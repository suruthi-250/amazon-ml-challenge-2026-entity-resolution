"""Text normalisation for business names and addresses.

Everything here is rule based and country agnostic: the same abbreviation
tables are applied to every record, whatever its country label.
"""
import re
import unicodedata

# --------------------------------------------------------------------------
# Indic script -> Latin transliteration
# All major Indic Unicode blocks share the Devanagari layout (ISCII heritage),
# so we fold every block onto Devanagari by offset and transliterate once.
# --------------------------------------------------------------------------
_INDIC_BLOCKS = [0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d",
    0x5D: "dh", 0x5E: "f", 0x5F: "y",
}
_VOW = {
    0x04: "a", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "l", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o",
    0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "l",
}
_MATRA = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x62: "l", 0x63: "l",
}
_VIRAMA = 0x4D
_INDIC_RE = re.compile(r"[ऀ-෿]")


def _fold(ch: str) -> int:
    """Return the offset (0..0x7F) of an Indic char within its block, or -1."""
    o = ord(ch)
    if 0x0900 <= o < 0x0980:
        return o - 0x0900
    for b in _INDIC_BLOCKS:
        if b <= o < b + 0x80:
            return o - b
    return -1


def has_indic(s: str) -> bool:
    return bool(s) and _INDIC_RE.search(s) is not None


def transliterate(s: str) -> str:
    out = []
    pending = False  # a consonant waiting for its vowel
    for ch in s:
        off = _fold(ch)
        if off < 0:
            if pending:
                out.append("a")
                pending = False
            out.append(ch)
            continue
        if off in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[off])
            pending = True
        elif off in _MATRA:
            out.append(_MATRA[off])
            pending = False
        elif off == _VIRAMA:
            pending = False
        elif off in _VOW:
            if pending:
                out.append("a")
            out.append(_VOW[off])
            pending = False
        elif off in (0x01, 0x02):      # candrabindu / anusvara
            if pending:
                out.append("a")
            out.append("n")
            pending = False
        elif off == 0x03:              # visarga
            if pending:
                out.append("a")
            out.append("h")
            pending = False
        elif 0x66 <= off <= 0x6F:      # digits
            if pending:
                out.append("a")
            out.append(str(off - 0x66))
            pending = False
        # nukta, stress marks, etc. are ignored
    if pending:
        out.append("a")
    t = "".join(out)
    # word-final schwa deletion ("sistama" -> "sistam") for longer words
    return re.sub(r"(?<=[a-z]{3}[^aeiou])a\b", "", t)


# --------------------------------------------------------------------------
# Latin clean-up
# --------------------------------------------------------------------------
_ZW_RE = re.compile(r"[​-‏⁠﻿]")
_INDIC_SEG_RE = re.compile(r"[ऀ-෿]+(?:[\s.]+[ऀ-෿]+)*")
_WORD_RE = re.compile(r"[0-9a-z]+")


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


# token -> canonical form (applied to both names and addresses)
COMMON_MAP = {
    "&": "and", "n": "and", "et": "and",
}

LEGAL_MAP = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prvt": "pvt", "pv": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc", "incorporation": "inc", "lnc": "inc", "inc.": "inc",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "company": "co", "co": "co", "cos": "co", "companies": "co", "cie": "co", "compagnie": "co",
    "llc": "llc", "llp": "llp", "pllc": "pllc", "plc": "plc", "lp": "lp", "ltee": "ltd",
    "pc": "pc", "pa": "pa", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl",
    "sa": "sa", "sci": "sci", "snc": "snc", "ei": "ei", "scop": "scop", "selarl": "selarl",
}
# Multi-token legal forms collapsed before tokenisation
_LEGAL_PHRASES = [
    # any dotted single-letter acronym: "e.u.r.l." -> "eurl", "l.l.c." -> "llc", "o.d." -> "od"
    (r"\b(?:[a-z]\s?\.\s?){2,}(?:[a-z]\b\.?)?", lambda m: re.sub(r"[\s.]", "", m.group(0)) + " "),
    (r"\bl\s*\.?\s*l\s*\.?\s*c\b", "llc"), (r"\bl\s*\.?\s*l\s*\.?\s*p\b", "llp"),
    (r"\bp\s*\.?\s*l\s*\.?\s*l\s*\.?\s*c\b", "pllc"), (r"\bp\s*\.\s*c\b", "pc"),
    (r"\bs\s*\.?\s*a\s*\.?\s*r\s*\.?\s*l\b", "sarl"), (r"\bs\s*\.\s*a\s*\.\s*s\s*\.?\s*u\b", "sasu"),
    (r"\bs\s*\.\s*a\s*\.\s*s\b", "sas"), (r"\bs\s*\.\s*a\b\.?", "sa"), (r"\bs\s*\.\s*c\s*\.\s*i\b", "sci"),
    (r"\bpvt\s*\.?\s*ltd\b", "pvt ltd"), (r"\bpra?\s*\.\s*li?\s*\.", "pvt ltd"),
    (r"\bo\s*\.\s*d\b\.?", "od"), (r"\bm\s*\.\s*d\b\.?", "md"), (r"\bd\s*\.\s*d\s*\.\s*s\b\.?", "dds"),
    (r"\bm\s*/\s*s\b", " "), (r"\bf\s*/\s*k\s*/\s*a\b", " fka "), (r"\ba\s*/\s*k\s*/\s*a\b", " aka "),
    (r"\bd\s*/\s*b\s*/\s*a\b", " dba "), (r"\bd\s*\.\s*b\s*\.\s*a\b\.?", " dba "),
]
_LEGAL_PHRASES = [(re.compile(p), r) for p, r in _LEGAL_PHRASES]

LEGAL = set(LEGAL_MAP.values()) | {"and"}
# Filler words the noise generator likes to add / drop around names
NAME_FILLER = {
    "the", "mr", "mrs", "ms", "dr", "m", "s", "www", "com", "net", "org", "in", "fr", "co", "of",
    "de", "du", "des", "la", "le", "les", "l", "d", "et", "and", "a",
    "center", "centre", "service", "services", "group", "holding", "holdings", "india", "france",
    "usa", "us", "enterprises", "ets", "etablissements",
    # honorifics prefixed by one source only
    "smt", "shri", "sri", "sh", "kum", "miss", "messrs",
}
_ALIAS_RE = re.compile(r"\b(?:aka|dba|fka|nee|formerly(?: known as)?|trading as|doing business as|"
                       r"also known as|t/a)\b\s*:?")
_ID_RE = re.compile(r"[(\[]?\s*\b(?:id|ref|reg)\s*(?:no|#)?\s*[:#]\s*\d+\s*[)\]]?")
_PHONE_RE = re.compile(r"\+?\d[\d \-]{7,}\d")
_URL_TAIL_RE = re.compile(r"\.(?:com|net|org|in|fr|co|biz|info|us)\b")

ADDR_MAP = {
    # street types (EN)
    # "St" is Street (EN) or Saint (FR): both collapse to one token
    "st": "st", "str": "st", "street": "st", "saint": "st", "rd": "road", "road": "road",
    "ave": "avenue", "av": "avenue", "avenue": "avenue", "avn": "avenue", "aven": "avenue",
    "dr": "drive", "drv": "drive", "drive": "drive", "ln": "lane", "lane": "lane",
    "pl": "place", "place": "place", "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard",
    "bvd": "boulevard", "boulevard": "boulevard", "ct": "court", "crt": "court", "court": "court",
    "cir": "circle", "circ": "circle", "circle": "circle", "trl": "trail", "tr": "trail", "trail": "trail",
    "hwy": "highway", "highway": "highway", "pkwy": "parkway", "pky": "parkway", "parkway": "parkway",
    "sq": "square", "square": "square", "terr": "ter", "terrace": "ter",
    "way": "way", "wy": "way", "cres": "crescent", "crescent": "crescent", "hts": "heights",
    "mt": "mount", "mtn": "mountain", "ft": "fort", "pt": "point", "cv": "cove", "xing": "crossing",
    "expy": "expressway", "fwy": "freeway", "aly": "alley", "loop": "loop", "run": "run", "pass": "pass",
    "cyn": "canyon", "vly": "valley", "vw": "view", "rdg": "ridge", "spg": "spring", "spgs": "springs",
    "lk": "lake", "lks": "lakes", "hl": "hill", "hls": "hills", "is": "island", "est": "estates",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    "rte": "route", "rt": "route", "cty": "county", "twp": "township", "cdp": "", "grnd": "grand",
    # FR
    "r": "rue", "rue": "rue", "ch": "chemin", "chem": "chemin", "chemin": "chemin",
    "all": "allee", "allee": "allee", "imp": "impasse", "impasse": "impasse", "rte": "route",
    "pass": "passage", "passage": "passage", "qu": "quai", "quai": "quai", "crs": "cours",
    "sq": "square", "res": "residence", "residence": "residence", "lot": "lotissement",
    "lotissement": "lotissement", "bis": "bis", "ter": "ter", "fg": "faubourg", "fbg": "faubourg",
    "prom": "promenade", "hameau": "hameau", "ham": "hameau", "zi": "zi", "za": "za",
    "ste": "ste", "sainte": "ste", "pte": "porte", "mal": "marechal", "q": "quai",
    "gen": "general", "gal": "general", "pdt": "president", "prof": "professeur",
    # IN
    "nr": "near", "opp": "opposite", "bldg": "building", "bldng": "building", "apt": "apartment",
    "apts": "apartment", "appt": "apartment", "flr": "floor", "fl": "floor", "flat": "flat",
    "sec": "sector", "sect": "sector", "sector": "sector", "ngr": "nagar", "nagar": "nagar",
    "mkt": "market", "mg": "mg", "stn": "station", "ind": "industrial", "indl": "industrial",
    "est": "estate", "extn": "extension", "ext": "extension", "vill": "village", "vil": "village",
    "po": "po", "ps": "ps", "dist": "district", "distt": "district", "tq": "taluk", "tal": "taluk",
    "chsl": "chs", "chs": "chs", "soc": "society", "mahal": "mahal", "colony": "colony", "col": "colony",
    # Indian city-name variants (old/new names, common spellings)
    "bombay": "mumbai", "poona": "pune", "madras": "chennai", "calcutta": "kolkata",
    "bengaluru": "bangalore", "banglore": "bangalore", "bengalore": "bangalore",
    "gurugram": "gurgaon", "cochin": "kochi", "calicut": "kozhikode", "mysuru": "mysore",
    "trivandrum": "thiruvananthapuram", "vizag": "visakhapatnam", "vishakhapatnam": "visakhapatnam",
    "vishakapatnam": "visakhapatnam", "baroda": "vadodara", "banaras": "varanasi", "benares": "varanasi",
    "keralam": "kl", "dl": "delhi", "belgaum": "belagavi", "mangalore": "mangaluru", "hubli": "hubballi",
    "ahmadabad": "ahmedabad", "kanpur": "kanpur", "allahabad": "prayagraj", "gauhati": "guwahati",
}
# tokens that carry no locating information
ADDR_DROP = {
    "unit", "ste", "suite", "no", "number", "num", "door", "h", "hno", "house", "null", "none",
    "na", "n/a", "nan", "box", "p", "o", "c", "of", "the", "de", "du", "des", "la", "le", "les", "l", "d",
    "and", "at", "", "addr", "address", "nº", "n°", "po box", "hn", "pmb",
}
_POBOX_RE = re.compile(r"\b(?:p\s*\.?\s*o\s*\.?\s*box|post\s+office\s+box|pmb|box)\s*#?\s*\d+\b")
_NUMSIGN_RE = re.compile(r"\bn\s*[\u00b0\u00ba]\s*")

_ORD_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
    "thirteenth": "13", "fourteenth": "14", "fifteenth": "15", "sixteenth": "16", "seventeenth": "17",
    "eighteenth": "18", "nineteenth": "19", "twentieth": "20", "premier": "1", "premiere": "1",
    "deuxieme": "2", "troisieme": "3",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "jammu and kashmir": "jk", "jammu kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py", "ladakh": "la",
}
FR_REGIONS = {
    "hauts de france": "hdf", "pays de la loire": "pdl", "nouvelle aquitaine": "naq",
    "ile de france": "idf", "auvergne rhone alpes": "ara", "grand est": "ges",
    "provence alpes cote d azur": "pac", "occitanie": "occ", "bretagne": "bre", "normandie": "nor",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "corse": "cor",
    # departements -> their region code
    "gironde": "naq", "landes": "naq", "dordogne": "naq", "lot et garonne": "naq",
    "pyrenees atlantiques": "naq", "charente maritime": "naq", "charente": "naq", "vienne": "naq",
    "deux sevres": "naq", "haute vienne": "naq", "creuse": "naq", "correze": "naq",
    "loire atlantique": "pdl", "maine et loire": "pdl", "sarthe": "pdl", "mayenne": "pdl", "vendee": "pdl",
    "nord": "hdf", "pas de calais": "hdf", "aisne": "hdf", "oise": "hdf", "somme": "hdf",
    "paris": "paris", "rhone": "ara", "bouches du rhone": "pac", "haute garonne": "occ",
}
_STATE_RE = None
_STATE_MAP = {}


def _state_re():
    global _STATE_RE, _STATE_MAP
    if _STATE_RE is None:
        for d in (US_STATES, IN_STATES, FR_REGIONS):
            for k, v in d.items():
                _STATE_MAP[k] = v  # same token as the raw abbreviation ("illinois" -> "il")
        keys = sorted(_STATE_MAP, key=len, reverse=True)
        _STATE_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")
    return _STATE_RE


_SPLIT_RE = re.compile(r"[^0-9a-z]+")
_ALNUM_RE = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
_ORDSUF_RE = re.compile(r"\b(\d+)\s*(?:st|nd|rd|th|er|e|eme|ere)\b")


def base_clean(s, lexicon=None):
    """lowercase, strip accents / zero-width chars, transliterate Indic scripts."""
    if not s:
        return ""
    s = _ZW_RE.sub("", s)
    if has_indic(s):
        def _seg(m):
            t = strip_accents(transliterate(m.group(0))).lower()
            if lexicon:
                t = _WORD_RE.sub(lambda w: lexicon.get(w.group(0), w.group(0)), t)
            return " " + t + " "
        s = _INDIC_SEG_RE.sub(_seg, s)
    s = strip_accents(s).lower()
    s = s.replace("&", " and ").replace("+", " and ").replace("ß", "ss").replace("æ", "ae").replace("œ", "oe")
    return s


def name_tokens(raw, lexicon=None):
    """Return (main_tokens, core_tokens) for a business name."""
    s = base_clean(raw, lexicon)
    if not s:
        return [], []
    if "|" in s:  # "Era Forum Private | www.eraforum.com"
        parts = [p for p in s.split("|") if p.strip()]
        s = max(parts, key=lambda p: (".com" not in p and "www" not in p, len(p))) if parts else s
    for rx, rep in _LEGAL_PHRASES:
        s = rx.sub(rep, s)
    m = list(_ALIAS_RE.finditer(s))
    if m:
        after = s[m[-1].end():]
        if after.strip():
            s = after
    s = _ID_RE.sub(" ", s)
    s = _PHONE_RE.sub(" ", s)
    s = _URL_TAIL_RE.sub(" ", s)
    s = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", s)  # H0RIZON -> HORIZON
    toks = [t for t in _SPLIT_RE.split(s) if t]
    toks = [LEGAL_MAP.get(COMMON_MAP.get(t, t), COMMON_MAP.get(t, t)) for t in toks]
    core = [t for t in toks if t not in LEGAL and t not in NAME_FILLER]
    if not core:
        core = [t for t in toks if t not in LEGAL] or toks
    return toks, list(dict.fromkeys(core))  # "Reify Reify Continental" -> once


def addr_tokens(raw, lexicon=None):
    if raw:
        raw = _NUMSIGN_RE.sub(" ", raw.lower())   # "N° 13" (numero), before accents are stripped
    s = base_clean(raw, lexicon)
    if not s:
        return []
    s = _POBOX_RE.sub(" ", s)                      # injected "PO Box 881" / "PMB 4445"
    s = _state_re().sub(lambda m: " " + _STATE_MAP[m.group(0)] + " ", re.sub(r"[\-_]", " ", s))
    s = _ORDSUF_RE.sub(r"\1", s)
    s = _ALNUM_RE.sub(" ", s)
    out = []
    for t in _SPLIT_RE.split(s):
        if not t:
            continue
        t = _ORD_WORDS.get(t, t)
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDR_MAP.get(t, t)
        if t in ADDR_DROP or not t:
            continue
        out.append(t)
    return out
