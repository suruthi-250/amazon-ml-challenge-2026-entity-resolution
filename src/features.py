"""Step 3: pairwise features for (Source-1, Source-2/3) candidate pairs."""
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

import textnorm as T

LEGAL_SET = sorted(T.LEGAL - {"and"})


def context_features(c):
    """Blocking-score context of each pair relative to the other candidates of
    the same right record and of the same Source-1 record."""
    c = c.sort(["rrow", "bscore"], descending=[False, True]).with_columns(
        r_best=pl.col("bscore").max().over("rrow"),
        r_next=pl.col("bscore").shift(-1).over("rrow"),
        r_ncand=pl.len().over("rrow").cast(pl.Int16),
        s_best=pl.col("bscore").max().over("s1row"),
        s_ncand=pl.len().over("s1row").cast(pl.Int16),
        s_rank=pl.col("bscore").rank("ordinal", descending=True).over("s1row").cast(pl.Int16),
    )
    return c.with_columns(
        gap_best=(pl.col("r_best") - pl.col("bscore")),
        gap_next=(pl.col("bscore") - pl.col("r_next").fill_null(0.0)),
        s_gap_best=(pl.col("s_best") - pl.col("bscore")),
    ).drop("r_best", "r_next", "s_best")


def _join_str(col):
    return pl.col(col).list.join(" ")


def _nums(col):
    return pl.col(col).list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")))


def _words(col):
    return pl.col(col).list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$")))


def _jacc(a, b):
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    uni = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(uni > 0).then(inter / uni).otherwise(0.0).cast(pl.Float32)


def _cover(a, b):
    """share of a's unique tokens found in b"""
    la = pl.col(a).list.unique().list.len()
    return pl.when(la > 0).then(pl.col(a).list.set_intersection(pl.col(b)).list.len() / la) \
             .otherwise(0.0).cast(pl.Float32)


def _grams(t):
    t = f" {t} "
    return {t[i:i + 3] for i in range(len(t) - 2)}


def _q3_jaccard(a_list, b_list):
    """character 3-gram Jaccard similarity (-1 when either side is empty)"""
    out = np.empty(len(a_list), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if not a or not b:
            out[i] = -1.0
            continue
        ga, gb = _grams(a), _grams(b)
        inter = len(ga & gb)
        out[i] = inter / (len(ga) + len(gb) - inter)
    return out


_IDF = {}


def name_idf(s1):
    """IDF of name core tokens over the Source-1 frame (cached per frame)."""
    key = (id(s1), s1.height)
    if key not in _IDF:
        n = s1.height
        df = s1.select(pl.col("n_core").list.unique().alias("t")).explode("t").drop_nulls("t") \
               .group_by("t").agg(df=pl.len())
        _IDF.clear()
        _IDF[key] = (df.with_columns(idf=(np.log(n + 1) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32))
                       .select("t", "idf"), float(np.log(n + 1)))
    return _IDF[key]


def _idf_features(x, s1):
    """frequency-aware name evidence: how rare are the shared / missing name words"""
    idf, unseen = name_idf(s1)
    x = x.with_row_index("_i")

    def agg(expr, name):
        e = x.select("_i", expr.alias("t")).explode("t").drop_nulls("t").join(idf, on="t", how="left") \
             .with_columns(pl.col("idf").fill_null(unseen))
        return e.group_by("_i").agg(pl.col("idf").sum().alias(name + "_sum"), pl.col("idf").max().alias(name + "_max"))

    a = agg(pl.col("n_core").list.unique(), "a")
    b = agg(pl.col("n_core_b").list.unique(), "b")
    sh = agg(pl.col("n_core").list.set_intersection(pl.col("n_core_b")), "sh")
    ms = agg(pl.col("n_core").list.set_difference(pl.col("n_core_b")), "ms")
    x = (x.join(a, on="_i", how="left").join(b, on="_i", how="left")
          .join(sh, on="_i", how="left").join(ms, on="_i", how="left").sort("_i"))
    z = pl.lit(0.0, pl.Float32)
    return x.with_columns(
        n_idf_cov_a=(pl.col("sh_sum").fill_null(0) / pl.col("a_sum")).fill_null(0).fill_nan(0).cast(pl.Float32),
        n_idf_cov_b=(pl.col("sh_sum").fill_null(0) / pl.col("b_sum")).fill_null(0).fill_nan(0).cast(pl.Float32),
        n_idf_max_shared=pl.col("sh_max").fill_null(z).cast(pl.Float32),
        n_idf_max_miss_a=pl.col("ms_max").fill_null(z).cast(pl.Float32),
    ).drop("_i", "a_sum", "a_max", "b_sum", "b_max", "sh_sum", "sh_max", "ms_sum", "ms_max")


def pair_features(c, s1, r, workers=-1):
    """c: pairs with rrow, s1row (+context cols). s1/r: frames with 'row' + token lists."""
    cols = ["row", "n_toks", "n_core", "a_toks"]
    x = (c.join(s1.select(cols), left_on="s1row", right_on="row", maintain_order="left")
          .join(r.select(cols + ["indic", "src"]), left_on="rrow", right_on="row", suffix="_b",
                maintain_order="left"))
    x = x.with_columns(
        na=_join_str("n_core"), nb=_join_str("n_core_b"),
        fa=_join_str("n_toks"), fb=_join_str("n_toks_b"),
        aa=_join_str("a_toks"), ab=_join_str("a_toks_b"),
        ca=pl.col("n_core").list.join(""), cb=pl.col("n_core_b").list.join(""),
        numa=_nums("a_toks"), numb=_nums("a_toks_b"),
        wa=_words("a_toks"), wb=_words("a_toks_b"),
        lga=pl.col("n_toks").list.eval(pl.element().filter(pl.element().is_in(LEGAL_SET))),
        lgb=pl.col("n_toks_b").list.eval(pl.element().filter(pl.element().is_in(LEGAL_SET))),
        ini=pl.col("n_toks").list.eval(pl.element().str.slice(0, 1)).list.join(""),
    )
    x = x.with_columns(
        numlong=pl.col("numa").list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True)).list.first())
    f = {}
    kw = dict(workers=workers, dtype=np.float32)
    na, nb = x["na"].to_list(), x["nb"].to_list()
    f["n_ratio"] = cpdist(na, nb, scorer=fuzz.ratio, **kw)
    f["n_tsort"] = cpdist(na, nb, scorer=fuzz.token_sort_ratio, **kw)
    f["n_tset"] = cpdist(na, nb, scorer=fuzz.token_set_ratio, **kw)
    f["n_partial"] = cpdist(na, nb, scorer=fuzz.partial_ratio, **kw)
    f["n_jw"] = cpdist(na, nb, scorer=distance.JaroWinkler.normalized_similarity, **kw)
    f["n_wratio"] = cpdist(na, nb, scorer=fuzz.WRatio, **kw)
    f["n_q3"] = _q3_jaccard(na, nb)
    del na, nb
    fa, fb = x["fa"].to_list(), x["fb"].to_list()
    f["nf_tsort"] = cpdist(fa, fb, scorer=fuzz.token_sort_ratio, **kw)
    del fa, fb
    ca, cb = x["ca"].to_list(), x["cb"].to_list()
    f["c_ratio"] = cpdist(ca, cb, scorer=fuzz.ratio, **kw)
    f["c_partial"] = cpdist(ca, cb, scorer=fuzz.partial_ratio, **kw)
    f["ini_ratio"] = cpdist(x["ini"].to_list(), cb, scorer=fuzz.ratio, **kw)
    del ca, cb
    aa, ab = x["aa"].to_list(), x["ab"].to_list()
    f["a_tset"] = cpdist(aa, ab, scorer=fuzz.token_set_ratio, **kw)
    f["a_tsort"] = cpdist(aa, ab, scorer=fuzz.token_sort_ratio, **kw)
    f["a_partial"] = cpdist(aa, ab, scorer=fuzz.partial_token_set_ratio, **kw)
    f["a_q3"] = _q3_jaccard(aa, ab)
    # house number (first number of each address): typo-tolerant similarity
    ha = x["numa"].list.first().fill_null("").to_list()
    hb = x["numb"].list.first().fill_null("").to_list()
    hs = cpdist(ha, hb, scorer=distance.Levenshtein.normalized_similarity, **kw)
    hs[np.array([not u or not v for u, v in zip(ha, hb)], dtype=bool)] = -1.0
    f["hn_sim"] = hs
    del aa, ab
    x = x.with_columns(
        n_jacc=_jacc("n_core", "n_core_b"),
        n_cov_a=_cover("n_core", "n_core_b"),
        n_cov_b=_cover("n_core_b", "n_core"),
        n_first_eq=(pl.col("n_core").list.first() == pl.col("n_core_b").list.first()).fill_null(False).cast(pl.Int8),
        n_len_a=pl.col("n_core").list.len().cast(pl.Int16),
        n_len_b=pl.col("n_core_b").list.len().cast(pl.Int16),
        lg_eq=(pl.col("lga").list.sort() == pl.col("lgb").list.sort()).cast(pl.Int8),
        lg_jacc=_jacc("lga", "lgb"),
        a_jacc=_jacc("a_toks", "a_toks_b"),
        a_cov_a=_cover("a_toks", "a_toks_b"),
        a_cov_b=_cover("a_toks_b", "a_toks"),
        w_cov_a=_cover("wa", "wb"),
        num_jacc=_jacc("numa", "numb"),
        num_cov_a=_cover("numa", "numb"),
        num_cov_b=_cover("numb", "numa"),
        num_first_eq=(pl.col("numa").list.first() == pl.col("numb").list.first()).fill_null(False).cast(pl.Int8),
        num_long_in=pl.col("numb").list.contains(pl.col("numlong")).fill_null(False).cast(pl.Int8),
        num_long_len=pl.col("numlong").str.len_chars().fill_null(0).cast(pl.Int8),
        num_any_a=(pl.col("numa").list.len() > 0).cast(pl.Int8),
        num_any_b=(pl.col("numb").list.len() > 0).cast(pl.Int8),
        a_len_a=pl.col("a_toks").list.len().cast(pl.Int16),
        a_len_b=pl.col("a_toks_b").list.len().cast(pl.Int16),
        indic_b=pl.col("indic").cast(pl.Int8),
        src_b=pl.col("src").cast(pl.Int8),
        hn_reldiff=pl.when((pl.col("numa").list.len() > 0) & (pl.col("numb").list.len() > 0)).then(
            (pl.col("numa").list.first().cast(pl.Float64, strict=False)
             - pl.col("numb").list.first().cast(pl.Float64, strict=False)).abs()
            / pl.max_horizontal(pl.col("numa").list.first().cast(pl.Float64, strict=False),
                                pl.col("numb").list.first().cast(pl.Float64, strict=False), pl.lit(1.0))
        ).otherwise(-1.0).fill_null(-1.0).cast(pl.Float32),
    )
    x = _idf_features(x, s1)
    feat = x.select(c.columns + FEATURES_POLARS).with_columns(
        **{k: pl.Series(v) for k, v in f.items()})
    # how this pair compares with the other candidates of the same right record
    rel = {}
    for k in REL_BASE:
        rel["rel_" + k] = pl.col(k) - pl.col(k).max().over("rrow")
    return feat.with_columns(**rel)


FEATURES_POLARS = ["n_jacc", "n_cov_a", "n_cov_b", "n_first_eq", "n_len_a", "n_len_b", "lg_eq", "lg_jacc",
                   "a_jacc", "a_cov_a", "a_cov_b", "w_cov_a", "num_jacc", "num_cov_a", "num_cov_b",
                   "num_first_eq", "num_long_in", "num_long_len", "num_any_a", "num_any_b", "a_len_a", "a_len_b",
                   "indic_b", "src_b", "hn_reldiff", "n_idf_cov_a", "n_idf_cov_b", "n_idf_max_shared",
                   "n_idf_max_miss_a"]
FEATURES_FUZZ = ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "nf_tsort", "c_ratio", "c_partial", "ini_ratio",
                 "a_tset", "a_tsort", "a_partial", "n_wratio", "n_q3", "a_q3", "hn_sim"]
# Entity-level context (s_ncand, s_rank, s_gap_best) is computed but deliberately not a model
# feature: its distribution shifts between train and test (test has ~30% more candidates per entity).
FEATURES_CTX = ["bscore", "pscore", "brank", "r_ncand", "gap_best", "gap_next"]
REL_BASE = ["n_tsort", "n_tset", "c_partial", "a_tset", "a_jacc", "num_jacc", "n_q3", "n_idf_cov_a"]
FEATURES = FEATURES_CTX + FEATURES_POLARS + FEATURES_FUZZ + ["rel_" + k for k in REL_BASE]
