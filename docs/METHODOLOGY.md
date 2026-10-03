# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Quantum
**Team Members:** Suruthi H (Team Lead), Giridharan H, Sherlin A
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We normalise noisy business names and addresses, including transliterating Indic scripts. Blocking then uses hashed sparse TF‑IDF retrieval: rare-token inverted-index products retrieve the top 20 Source‑1 neighbours of every Source‑2/3 record, and a full-cosine re‑rank keeps at most 3. This gives **~6–8 candidates per Source‑1 entity** with a **97.0 % pair-recall ceiling**, and it never forms the |S1|×|S2+S3| cross product. A cross‑fitted gradient‑boosted matcher then scores each pair. A second stage re-scores each pair in the context of the other candidates of the same record. Finally, a one‑to‑one assignment gives each Source‑2/3 record to at most one entity, with a threshold tuned for macro F0.5 on held‑out entities.

Key innovations:
- **Cross tokens** (name word × address token) make retrieval work even when every individual token is common.
- **A learned, unsupervised noise vocabulary** removes words injected by the noise process.
- **Shift-robust modelling:** features whose distribution differs between train and test (entity-level candidate counts) are deliberately excluded. This was learned the hard way; see §5.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA on the training data)

| | Train | Test |
|---|---|---|
| Source 1 | 2,206,821 (US 1.32 M, India 0.88 M) | 1,732,544 (US 0.66 M, India 0.81 M, **France 0.26 M**) |
| Source 2 / Source 3 | 5.03 M / 5.29 M | 4.89 M / 5.08 M |
| Labelled links | 7,638,365 | – |

Key findings:

* **Every Source‑2/3 record belongs to at most one Source‑1 entity.** The 7.64 M labelled links contain 7.64 M distinct right‑side IDs. This turns matching into an *assignment* problem: each right record picks its best entity or none.
* 26 % of Source‑2/3 records match nothing (distractors). 5.6 % of Source‑1 entities are singletons. Linked entities have 3.46 matches on average (up to 11). Country labels never disagree within a true pair.
* **Names:**
  * abbreviations and legal-form variants (Pvt/Private, L.L.C., S.A.R.L., E.U.R.L.)
  * word transpositions and character typos (`H0RIZON`, `Prmefum`)
  * honorifics (`Smt`, `Shri`, `Mr`)
  * appended IDs (`(ID: 40117)`) and phone numbers
  * websites (`macillusionsprivate.com`, `#Modernconsultancy`) and initials (`DNGDP`)
  * alias constructions where the true name follows a marker (`X aka Y`, `X DBA: Y`, `X formerly: Y`, `X née Y`, `X doing business as Y`)
  * rebrands (a completely different name at an identical address)
  * **names written in 9 Indic scripts** (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi)
* **Injected name words:** measured as Source‑2/3 frequency vs Source‑1 frequency per country. Examples by country:
  * US: *Midtown, Northside, Greater, Riverside, Eastgate …*, with ratios around 30×.
  * India: *Overseas, Infratech*, shop-type suffixes such as *Stores, Sweets* and their transliterations.
  * France (test inputs only): *Participations, Distribution, Associés, Développement, Groupe, International*.
* **Addresses:**
  * street-type abbreviations; "St" meaning Street or Saint
  * ordinals written as words (`TWELFTH` = `12th`)
  * component reordering
  * missing house numbers, cities or whole addresses (≈3 %)
  * injected `PO Box` / `PMB` numbers
  * `N°` numbering
  * zero padding (`0023574`)
  * states given in full, abbreviated or in native script
  * Indian city aliases (Bombay/Mumbai, Poona/Pune, Gurugram/Gurgaon, Keralam)
  * French départements (Gironde, Nord) replacing régions
* **Many Source‑1 entities share the same name at different addresses** (chains, and generated names like "Digital Consultancy Pvt Ltd"). The address is therefore essential for disambiguation, and a record with an empty address is often inherently ambiguous.

### 2.2 Solution Strategy

**Approach Type:** Blocking (sparse retrieval + re‑rank) → two‑stage gradient‑boosted classifier → constrained one‑to‑one assignment.
**Core Innovation:**
- Cross-token TF‑IDF retrieval with a rare-token inverted index and full-cosine re‑ranking.
- An unsupervised injected-noise vocabulary.
- A stage-B model restricted to shift-robust, record-level context.

Country is treated as an **open set of labels**. It is only used to restrict comparisons to records with the same label, and it is never a model feature. France therefore flows through the same code path with no special casing.

Normalisation (`textnorm.py`, `prep.py`) is applied identically to every source:

1. **Clean-up:** Unicode NFKD, accent and zero-width-character removal, lower-casing.
2. **Indic transliteration:** all Indic Unicode blocks share the Devanagari layout, so every block is folded onto Devanagari by code-point offset and transliterated once, including consonant/matra/virama handling and final-schwa deletion.
3. **Lexicon learned from the training ground truth:** a co-occurrence association between transliterated tokens and the English tokens of the matched Source‑1 names/addresses (e.g. `sistams→systems`, `praivet→pvt`, `eksaports→exports`, `madhy pradesh→mp`). 678 entries; nothing external.
4. **Names:**
   - dotted-acronym collapse (`e.u.r.l.`→`eurl`)
   - legal-form canonicalisation (EN/IN/FR) and removal from the *core* name
   - alias handling: keep the part after `aka/dba/fka/formerly/née/doing business as`
   - ID/phone/URL stripping, `0→o` inside words, de-duplication of repeated words
5. **Addresses:**
   - EN/FR/IN abbreviation tables; ordinal words → numbers; `12th`→`12`; leading zeros stripped
   - US states, Indian states and French régions/départements → codes; Indian city aliases unified
   - `PO Box/PMB n` and `N°` removed; letter–digit splitting (`A24`→`a 24`)
6. **Injected-noise vocabulary (unsupervised):** per split and country label, name tokens at least 3.5× more frequent in Source 2/3 than in Source 1 (≥300 occurrences) are dropped from the core name. It uses only unlabelled inputs and is recomputed for test, so France gets its own list automatically.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used.** Each record becomes an L2‑normalised TF‑IDF vector over *hashed* (u64), field‑prefixed tokens, with IDF computed on Source 1 per country label:

| token | example | purpose |
|---|---|---|
| `n\|word` | `n\|charities` | name core words |
| `p\|w1_w2` | `p\|charities_heart` | unordered pairs of the first 5 name words (order-insensitive, rarer than single words) |
| `c\|xxxxxx` | `c\|macill` | first 6 characters of the space-less name (websites, run-together names) |
| `a\|tok` | `a\|mississippi` | address tokens |
| `h\|num_word` | `h\|1658_mississippi` | house number + following word |
| `x\|word\|tok` | `x\|digital\|gurgaon` | **cross tokens**: first two name words × every address token |

**Two-pass retrieval** (`blocking.py`), for each Source‑2/3 record against Source‑1 records with the same country label:

1. **Retrieve.** Compute a partial cosine over *rare* tokens only, i.e. Source‑1 document frequency ≤ 1000 (≤ 300 for cross tokens). This is done through an inverted index as chunked sparse matrix products (`R_chunk @ S1_rareᵀ`) in 6 worker processes, keeping the top 20. Posting lists are capped, so the cost is O(Σ posting-list lengths), not O(|S1|·|S2+S3|).
2. **Re‑rank.** Add the common-token part of the dot product for those 20 (full cosine) and keep the top 5 (floor 0.05).
3. **Final candidate filter** (`match.select_candidates`), the exact set the model scores: rank < 3, cosine ≥ 0.10 and ≥ 0.5 × the record's best cosine.

Because every right record belongs to at most one entity, retrieving a few Source‑1 neighbours *per right record* keeps the candidate set tiny.

**Candidate pairs generated:**

| | pairs | per Source‑1 entity | pair recall (true links kept) |
|---|---|---|---|
| Train (raw top‑5) | 51.1 M | 23.2 | – |
| **Train (final)** | **13.63 M** | **6.18** | **0.970** |
| **Test (final, `candidate_pairs.tsv`)** | **13.88 M** | **8.01** | – |

Test has more Source‑2/3 records per entity than train (5.8 vs 4.7), hence the larger per-entity count. Reduction ratio against the full cross product: 1 − 13.88 M / (1.73 M × 9.97 M) ≈ **1 − 8·10⁻⁷**. The raw Source‑2/3 → Source‑1 retrieval runs in ~40 min for all 10 M test records on a 4‑core laptop.

**How we ensured true matches were not lost.** Every design choice was measured on 60k‑record samples as the recall of the true Source‑1 entity among each record's neighbours:

| setting | US @1 | US @3 | India @1 | India @3 |
|---|---|---|---|---|
| words/pairs/prefix/address, rare-index only | 0.925 | 0.956 | 0.879 | 0.915 |
| + full-cosine re‑rank of top 20 | 0.946 | 0.966 | 0.905 | 0.932 |
| + cross tokens (name × address) | **0.962** | **0.976** | **0.953** | **0.970** |

- Tightening the cross-token cap to 300 and dropping partial scores < 0.02 made retrieval 3.2× faster with −0.1 pt recall.
- Normalisation fixes (states, city aliases, `St`/Saint, PO boxes, noise words) lifted the final-filter recall from 0.934 to 0.970 at a *smaller* candidate set (6.2 vs 8.6 per entity).
- Most remaining misses are records with an empty address whose name is shared by several Source‑1 entities, which no method can resolve.

---

## 4. Matching Model

**Features used** (`features.py`, 59 pairwise features in the final v5 model, rapidfuzz `cpdist` multithreaded):

- **Name:**
  - core-name `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, Jaro–Winkler; full-name token sort ratio
  - space-less name `ratio` / `partial_ratio` (websites)
  - initials of Source‑1 name vs space-less right name
  - token Jaccard and coverage in both directions; first-token equality; token counts
  - legal-form equality / Jaccard
  - *(v5)* `WRatio`; character 3‑gram Jaccard of the core names
  - *(v5)* **frequency-aware name evidence** (IDF over Source‑1 core-name words): IDF-weighted coverage of the Source‑1 name by the right name and vice versa, the rarest shared word, and the rarest Source‑1 word *missing* on the right. The last one is what separates sibling businesses such as `Municipaux Sportive` vs `Municipaux Musique` at the same address.
- **Address:**
  - `token_set_ratio`, `token_sort_ratio`, `partial_token_set_ratio`
  - token Jaccard and coverage; word-only coverage
  - numbers: Jaccard, coverage in both directions, first-number equality, "longest Source‑1 number appears on the right" and its length
  - empty-address and length indicators
  - *(v5)* character 3‑gram Jaccard of the address; **house-number similarity**: Levenshtein similarity of the two first numbers (tolerates `3614` vs `2614`) and relative numeric difference (separates `591` vs `593` from `7` vs `70`)
- **Blocking context:**
  - full and partial cosine, rank, number of candidates of the right record and of the entity
  - gap to the right record's best / next candidate
  - Entity-level context (candidates per entity, rank within the entity) is computed but *not* used: test has ~30 % more candidates per entity than train (median 9 vs 7, p90 20 vs 13), so these features do not transfer.
  - "relative" features: each key similarity (incl. 3‑gram and IDF coverage) minus its maximum over the record's candidates
- **Other:** Indic-script flag, source (S2/S3).

**Model type (final v7T = v5 + transformer):**
0. **Cross-encoder** (`gpu_crossencoder.py`, GPU): the pretrained `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache‑2.0, ~22M parameters, well under the 8B limit) is fine-tuned for 1 epoch.
   - **Training data:** 2.4M labelled candidate pairs from the *non-validation* training records (1.2M matches / 1.2M non-matches).
   - **Input:** each pair as text, `name: … | addr: …` for both records, in the same normalised form v5 uses.
   - **Output:** it scores every validation and test candidate pair, producing pT.
   - **Where it helps:** in the uncertain band (v5 score 0.30–0.74), it separates true from false links far better than v5 (AUC 0.79 vs 0.64), so it carries genuinely new evidence.
1. **Stage A:** `HistGradientBoostingClassifier` (127 leaves, up to 1500 trees, early stopping). It is cross‑fitted: two models are trained on disjoint halves of the Source‑2/3 records, so every training pair gets an out‑of‑fold probability pA. On test, the two models' probabilities are averaged.
2. **Stage B:** in v7T, pT plus its rank and margin among the record's candidates are added to the stage‑B inputs listed below (`combine_transformer.py`). Stage B is a small gradient-boosted model (31 leaves) on **right-record context**: pA; the rank of pA among the record's candidates; the best competing candidate's pA; margin; sum and count of the other candidates' pA; blocking rank, cosine and candidate count. It is fitted on the validation pairs, whose averaged pA is distributed exactly like test pA, with 2-fold cross-fitting for threshold tuning.
3. **Unseen country labels:** the cross-encoder was trained only on the country labels present in the training data (US, India). On the unlabelled France records it proved over-confident: it would have re-added ~42k look-alike sibling links, half of them exactly the links v5 had correctly dropped. So for any country label not seen in training, the v5 decision is kept unchanged. This rule is derived from the training data, not a hard-coded country list.
4. **Decision:** constrained one‑to‑one assignment. Each Source‑2/3 record is linked to its highest-scoring candidate if pB ≥ threshold; otherwise it is left unlinked. Source‑1 entities with no linked record are output as singletons (empty list).

All models are trained from scratch on the provided training data (scikit‑learn, BSD‑3). No pretrained model, external data, API or geocoder is used.

**Threshold selection method:** grid search of the pB threshold maximising **macro F0.5 on held‑out Source‑1 entities**. The held-out set is a hash split, 1/10 of all entities, with every right record touching them excluded from training.

---

## 5. Results & Error Analysis

Validation = 220,565 held‑out Source‑1 entities (`python evaluate.py` reproduces this):

| model | macro F0.5 | precision | recall | public LB |
|---|---|---|---|---|
| v1: stage A + right-context stage B | 0.9734 | 0.993 | 0.939 | 0.964 |
| v2: + more data/trees, source feature, entity/peer-context stage B | 0.9792 | 0.995 | 0.950 | **0.949** |
| v3: v2 stage A + right-context stage B | 0.9750 | – | – | not submitted |
| v4: v3 without entity-level context features | 0.9743 | 0.994 | 0.938 | 0.966 |
| **v5 (final): v4 + 3‑gram, WRatio, house-number similarity, name IDF evidence** | **0.9783** | **0.995** | **0.946** | **0.971** |
| v6: v5 + Monge–Elkan / SoftTF‑IDF name evidence | 0.9783 | – | – | rejected (no gain) |
| **v7T (final): v5 + fine-tuned cross-encoder score in stage B (US/India); v5 unchanged for unseen country labels** | **0.9838** | **0.997** | **0.956** | **0.978** |

Final model v5 per country: US F0.5 0.980 (P 0.996, R 0.949); India 0.975 (P 0.993, R 0.941). 12,170 of 12,446 singletons (97.8 %) were correctly left empty; 722,417 of 725,834 predicted links are correct (pair precision 0.9953).

**v5 robustness check before submission.** v5 was compared with v4 through the complete pipeline on 4 disjoint entity-level validation quarters:
- **Every quarter improved**, by +0.38 to +0.42 points. A paired bootstrap over 220,565 entities gives **+0.40 points, 95 % CI [+0.38, +0.43]**.
- False links fell by 24 % while predicted links rose only 0.6 %.
- **Corruption tests:** 30,000 validation records, each with one controlled corruption, sent through normalisation, blocking and matching.

| corruption | v4 recall | v5 recall |
|---|---|---|
| one digit of the house number dropped or substituted | 0.739 | **0.825** |
| one-character typo in one name word | 0.775 | **0.873** |
| name words shuffled | 0.950 | 0.963 |
| one name word removed | 0.882 | 0.885 |
| address removed | 0.483 | 0.482 |
| none | 0.973 | 0.981 |

The public leaderboard confirmed the local result: v4 0.966 → v5 0.971.

**v7T check before submission (vs v5):**
- **Validation:** 0.97831 → **0.98378** (+0.55 points). Split-half tests gave +0.54 / +0.56, and all 4 holdouts improved by +0.51 to +0.58.
- **Paired bootstrap:** +0.55 points, 95 % CI [+0.52, +0.57].
- **Errors:** wrong links 3,417 → 2,081 (−39 %); recall 0.946 → 0.956.
- **Test link counts:** +0.6 % (US) and +1.0 % (India), in line with validation.
- **Public leaderboard: 0.971 → 0.978.**

**What we tried after v5 and rejected on evidence:**
- **Segment-specific thresholds** (by address situation × name identity): +0.003 / −0.005 points in split-half tests, i.e. noise.
- **A margin-based "recovery gate"** for near-threshold true matches: the uncertain records (score 0.50–0.74) are only 50–68 % true, even at high margins. The best gate lost 0.013 points on held-out entities.
- **v6** (Monge–Elkan and SoftTF‑IDF word-level fuzzy name evidence): validation 0.97829 vs 0.97831. v5's 3‑gram and edit-distance features already carry this information.
- **Unit-number and whole-name-uniqueness features:** no bucket separable enough to learn from. Units are written inconsistently across sources; the uncertain band is 44–65 % true in every uniqueness bucket.

- **Loss breakdown (v1):**
  - 75 % from entities that are partially right, almost always a *missed* link
  - 20 % from linked entities predicted empty
  - 5 % from singletons that received a false link
- **Common false positives (wrong merges):** chain/franchise entities with identical names at nearby addresses; right records with an empty address that picked the wrong same-name entity; generic Indian names ("Shree", "Sky Investments") where the house number was corrupted.
- **Common false negatives (missed matches):** empty or heavily truncated addresses combined with a common name; rebrands with an unrelated name (`Synjaxnyla Labs` at the entity's exact address); initials and run-together website names; Indic-script names whose words are outside the learned lexicon; typos in both the house number and the name.

**The v2 lesson: train/test shift.** v2 added an entity-level "peer" stage: how similar a record is to the confident records already linked to the same entity. It gained +0.58 points on validation but *lost* 2.4 points on the leaderboard. Diagnosis on the unlabelled test predictions showed the cause:
- Test contains many look-alike **sibling businesses**: the same name on the same street at a nearby house number (e.g. `591` vs `593 Gipper Way`).
- Stage A scored them low (median 0.15), but the peer stage pushed ~277k of them over the threshold, because they resemble the entity's genuine matches.
- Test also has ~30 % more candidates per entity than train, so every entity-level feature is shifted.

We therefore dropped all entity-level context. Stage B uses only record-level context (v3), and stage A no longer uses candidates-per-entity features (v4). The remaining validation-to-leaderboard gap (0.978 vs 0.971) is consistent with France being absent from training and with the denser test distribution.

**Where the remaining error is (v5, validation).** 94.6 % of true links are found.
- **3.0 % are never candidates.** Half of these are records with no address whose name is shared by several Source‑1 businesses; nothing in the data identifies the right one.
- **2.0 % are the top candidate but below the threshold.** These are genuinely uncertain cases (~60/40).
- **0.02 % lose to another business.** Assignment errors are negligible.

---

## 6. Conclusion

Our blocking combines cross tokens, rare-token index retrieval and a full-cosine re‑rank. It keeps 97 % of true links with only ~6–8 candidates per entity and runs on a laptop in minutes per million records. A cross‑fitted gradient-boosted matcher (frequency-aware name evidence, typo-tolerant house-number comparison) combined with a small fine-tuned cross-encoder in a record-level context stage, plus one‑to‑one assignment, reaches 0.984 macro F0.5 on held‑out entities with 99.7 % precision, and **0.978 on the public leaderboard**.

The biggest lessons:
- Normalisation errors (noise words, `N°`, Saint/Street) cost more than model capacity.
- The "each record belongs to one entity" structure is worth exploiting at every stage.
- A validation gain is not a leaderboard gain when features depend on data density. Check feature distributions on the unlabelled test inputs before trusting them.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its `README.md`; `requirements.txt` pins Python 3.13 packages):

| file | step |
|---|---|
| `src/config.py` | paths (override with `ER_DATA_ROOT`, `ER_WORK_DIR`, `ER_OUTPUT_DIR`), parallelism |
| `src/textnorm.py` | normalisation rules and Indic transliteration |
| `src/prep.py` | 1: tokenise all sources, learn lexicon, learn noise vocabulary → parquet |
| `src/blocking.py` | 2: hashed TF‑IDF, rare-token retrieval, full-cosine re‑rank → `cands_raw.parquet` |
| `src/features.py` | pairwise features |
| `src/match.py` | 3/4: final candidate filter, stage A/B models, assignment, threshold, TSV writers |
| `src/evaluate.py` | precision / recall / F0.5 on the validation split |
| `src/check_score.py` | pre-upload check: validation score + link-count shift vs the last scored file |
| `src/error_archetypes.py` | recall / error share per error type (name / address situation) |
| `src/segment_thresholds.py` | split-half test of segment-specific thresholds (negative result) |
| `src/rethreshold.py` | re-write the submission at another threshold from saved test scores |
| `src/export_pairs.py` | exports candidate pairs as compact text tables for the GPU step |
| `src/gpu_crossencoder.py` | **GPU step** (Kaggle T4): fine-tunes the cross-encoder and scores validation + test pairs |
| `src/combine_transformer.py` | adds the cross-encoder score to stage B, applies the unseen-country fallback, writes the final submission |
| `src/gt.py` | ground-truth helpers |
| `src/run_all.py` | **entry point**: runs all steps and the official validator |

Reproduce:
1. `pip install -r requirements.txt`, then `cd src && python run_all.py`. This writes the v5 files (≈4 h on a 4‑core/16 GB laptop).
2. `python export_pairs.py`, then run `gpu_crossencoder.py` on a GPU (≈1.5 h on a T4; needs `torch` and `transformers`).
3. `python combine_transformer.py` writes the final v7T `matching_results.tsv`.

All randomness is seeded.

### B. Additional Results

Validation split: 220,565 held-out entities (US 132,540 / India 88,025).

| version | stage A val F0.5 | final val F0.5 | US | India | test links |
|---|---|---|---|---|---|
| v1 | 0.9727 | 0.9734 | 0.976 | 0.970 | 5,747,842 |
| v2 | 0.9744 | 0.9792 | 0.981 | 0.977 | 5,983,300 |
| v3 | 0.9744 | 0.9750 | 0.977 | 0.972 | 5,757,224 |
| v4 | 0.9737 | 0.9743 | 0.976 | 0.971 | 5,722,832 |
| **v5 (final)** | **0.9777** | **0.9783** | **0.980** | **0.975** | **5,715,987** |
| v6 (rejected) | 0.9777 | 0.9783 | 0.980 | 0.975 | – |

| **v7T (final)** | – | **0.9838** | – | – | **5,757,142** |

Public leaderboard history: v1 0.964 → v2 0.949 (entity-level peer stage; did not transfer) → v4 0.966 → v5 0.971 → **v7T 0.978**.
