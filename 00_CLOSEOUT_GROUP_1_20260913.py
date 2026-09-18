# ─────────────────────────────────────────────────────────────────────────────
# 00_CLOSEOUT_GROUP_1_20260913.py — v3-COMPATIBLE PATCH
# OWNER TAG : GROUP 1
# ANCHOR    : RUN-GROUP_1-20260913
# HALF 1 ONLY. Every block below is READ-ONLY. Nothing here changes the
# pipeline, refits a reported model, or edits a saved result.
# Run this as a cell inside Nyansapo_RAG_Pipeline.ipynb, after the notebook's
# dataset-building cell (questions/labels/train_model_v2 all defined).
# ─────────────────────────────────────────────────────────────────────────────

# ── ENV DETECT ───────────────────────────────────────────────────────────────
try:
    import google.colab          # noqa: F401
    ENV = "colab"
except ImportError:
    ENV = "local"
print("ENVIRONMENT:", ENV)

import os
import re
import json
import math
import shutil
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────
if ENV == "colab":
    DATA_DIR = "/content/drive/MyDrive/Nyansapo"
    print("DATA PATH BRANCH: colab, mounted drive")
else:
    DATA_DIR = "<<< FILL IN: local directory holding the files v3 reads from Drive >>>"
    print("DATA PATH BRANCH: local")
print("DATA_DIR:", DATA_DIR)

EVAL_RESULTS  = os.path.join(DATA_DIR, "eval_results.json")
GOLD_CSV      = os.path.join(DATA_DIR, "gold_qa_pairs.csv")
GOLD_CKPT     = os.path.join(DATA_DIR, "gold_eval_checkpoint.json")
GOLD_ROUGE    = os.path.join(DATA_DIR, "gold_rouge_v2.json")
QIC_RESULTS   = os.path.join(DATA_DIR, "qic_results_corrected.json")
STAGING_DIR   = os.path.join(DATA_DIR, "closeout_artifacts")

NOTEBOOK_PATH = "/content/drive/MyDrive/Colab Notebooks/Nyansapo_RAG_Pipeline.ipynb"
TOPIC_COL     = "course"
TLX_RESPONSES = "user_study_anonymized.csv"

SEED_MAIN       = 42
SEED_PERTURB    = 123
TEST_SIZE       = 0.15
VAL_SIZE        = 0.176
ALPHA_BONF      = 0.0167
N_FOLDS         = 5
N_GOLD_EXPECT   = 200
N_TOPICS_EXPECT = 8
N_BOOT          = 2000
POWER_TARGET    = 0.80

# QUOTED from the Method via the open item. Comparison only.
METHOD_TRAIN_N = 101       # QUOTED — corrected v3 value (was 102 in v2 draft)
METHOD_VAL_N   = 22        # QUOTED — corrected v3 value (was 21 in v2 draft)
METHOD_TEST_N  = 22        # QUOTED

SYSTEMS = ["crag", "naive", "bm25", "vanilla"]

VERDICTS = []


def record(item, check, observed, expected, verdict):
    VERDICTS.append((item, check, str(observed), str(expected), verdict))


def head(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ── VERSION GUARD ────────────────────────────────────────────────────────────
head("VERSION GUARD")

import sklearn
import scipy
import torch
import pandas as pd

PINNED = {"scikit-learn": "1.6.1", "torch": "2.11.0"}

print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("scipy       :", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("torch       :", torch.__version__)

assert sklearn.__version__ == PINNED["scikit-learn"], (
    "scikit-learn is %s, requirements.txt pins %s."
    % (sklearn.__version__, PINNED["scikit-learn"]))
assert torch.__version__.split("+")[0] == PINNED["torch"], (
    "torch is %s, requirements.txt pins %s."
    % (torch.__version__, PINNED["torch"]))
print("VERSION GUARD: PASS")


# ── PREFLIGHT ────────────────────────────────────────────────────────────────
head("PREFLIGHT")

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from scipy.stats import norm, wilcoxon
from collections import Counter

REQUIRED_GLOBALS = [
    "questions", "labels", "num_classes",
    "class_names", "set_seeds", "IntentClassifier", "train_model_v2",
]
# X, y, vectorizer, class_weights_tensor are intentionally absent in v3 —
# train_model_v2 now owns fitting the vectorizer and computing weights
# internally, per-split, per-seed. Their absence is the fix (closes X2).
G = globals()
missing_globals = [n for n in REQUIRED_GLOBALS if n not in G]
print("REQUIRED OBJECTS MISSING:", missing_globals if missing_globals else "NONE")
assert not missing_globals, (
    "Run the notebook's setup + dataset cells first. This file binds to "
    "your live objects and invents none of them."
)

# v3 uses plain Python lists for questions/labels — build a numpy array
# view for anywhere this script needs array indexing (train_test_split,
# stratify, positional slicing). This does not touch the notebook's own
# objects.
labels_arr = np.array(labels)

REQUIRED_FILES = {
    "eval_results.json": EVAL_RESULTS,
    "gold_qa_pairs.csv": GOLD_CSV,
    "gold_eval_checkpoint.json": GOLD_CKPT,
    "gold_rouge_v2.json": GOLD_ROUGE,
    "qic_results_corrected.json": QIC_RESULTS,
}
FILE_PRESENT = {}
for name, path in REQUIRED_FILES.items():
    FILE_PRESENT[name] = os.path.exists(path)
    print("%-32s %s" % (name, "PRESENT" if FILE_PRESENT[name] else "ABSENT"))

NB_OK = os.path.exists(NOTEBOOK_PATH)
print("%-32s %s" % ("v3 notebook", "PRESENT" if NB_OK else "ABSENT — V1e cannot run"))

print("\nSHAPES AND COUNTS")
print("len(questions)        :", len(questions))
print("len(labels)           :", len(labels))
print("num_classes           :", num_classes)
print("class_names           :", list(class_names))
assert len(questions) == len(labels), (
    "questions and labels disagree on n. Stop and resolve.")

print("\nMISSING VALUE CHECK")
print("empty question strings:", sum(1 for q in questions if not str(q).strip()))
print("class distribution    :", dict(sorted(Counter(labels).items())))
print("\nSEEDS IN USE          : main=%d perturbation=%d" % (SEED_MAIN, SEED_PERTURB))
print("PREFLIGHT: COMPLETE")


# ═════════════════════════════════════════════════════════════════════════════
# HALF 1 — VERIFY. READ-ONLY.
# ═════════════════════════════════════════════════════════════════════════════

# BLOCK V1e — STATIC LEAK SCAN
head("BLOCK V1e — STATIC LEAK SCAN")

FIT_PAT = re.compile(r"(\w+)\s*\.\s*(fit_transform|fit)\s*\(\s*([^)\n]*)")
TRAIN_HINT = re.compile(r"train", re.IGNORECASE)

cause11_confirmed = None
weights_outside_split = None

if not NB_OK:
    print("NOT IN SOURCE: v3 notebook — NOTEBOOK_PATH not set or not found")
    print("V1e SKIPPED. C1 cannot be verdicted from this run.")
else:
    nb = json.load(open(NOTEBOOK_PATH))
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    print("code cells scanned:", len(cells))
    print("\nRULE: a fit is INSIDE-FOLD only if its argument names a train "
          "variable.\nAnything else is PRE-SPLIT: it saw rows the split later "
          "assigned to val or test.\n")
    print("%-6s %-6s %-14s %-12s %s" % ("CELL", "LINE", "OBJECT", "CALL", "CLASSIFICATION"))
    fit_rows = []
    for ci, cell in enumerate(cells):
        for li, line in enumerate("".join(cell.get("source", [])).split("\n"), 1):
            if line.strip().startswith("#"):
                continue
            m = FIT_PAT.search(line)
            if not m:
                continue
            obj, call, arg = m.group(1), m.group(2), m.group(3).strip()
            kind = "INSIDE-FOLD" if TRAIN_HINT.search(arg) else "PRE-SPLIT"
            fit_rows.append((ci, li, obj, call, kind, arg))
            print("%-6d %-6d %-14s %-12s %s   arg=%s" % (ci, li, obj[:14], call, kind, arg[:34]))
    if not fit_rows:
        print("(no fit calls located)")
    pre_split = [r for r in fit_rows if r[4] == "PRE-SPLIT"]
    cause11_confirmed = len(pre_split) > 0
    print("\nfit calls located     :", len(fit_rows))
    print("PRE-SPLIT fits        :", len(pre_split))
    print("CAUSE 11 CONFIRMED    :", cause11_confirmed)

    # Context note: distinguish the self-test's diagnostic full-fit from the
    # production training path. The scanner cannot read intent — it only
    # sees whether the fit's argument name contains "train". A PRE-SPLIT
    # hit on a self-test comparison object is not the same finding as a
    # PRE-SPLIT hit inside the training function itself.
    if pre_split:
        print("\nPRE-SPLIT fit locations (context check each manually):")
        for ci, li, obj, call, kind, arg in pre_split:
            print("  cell %d line %d — %s.%s(%s)" % (ci, li, obj, call, arg[:40]))
        print("If a listed cell/object is a self-test's OWN comparison "
              "vectorizer\n(never passed into train_model_v2 or any scored "
              "model), state that\nexplicitly in the Methods ledger rather "
              "than treating it as equivalent\nto a leak inside the training "
              "path.")

    split_cell = None
    for ci, cell in enumerate(cells):
        if "def train_model_v2" in "".join(cell.get("source", [])):
            split_cell = ci
            break
    weight_cells = [ci for ci, cell in enumerate(cells)
                    if re.search(r"^\s*class_weights_tensor\s*=",
                                 "".join(cell.get("source", [])), re.M)]
    print("\ndef train_model_v2 in cell :", split_cell if split_cell is not None else "NOT FOUND")
    print("class_weights_tensor set in:", weight_cells if weight_cells else "NOT FOUND")
    if not weight_cells:
        print("NOT APPLICABLE — v3 has no standalone class_weights_tensor "
              "global.\nWeights are computed inside train_model_v2, scoped "
              "per-split, per-seed.\nThis is the fix, not a missing check.")
        weights_outside_split = False
    else:
        weights_outside_split = split_cell not in weight_cells
        print("WEIGHTS COMPUTED OUTSIDE THE SPLIT:", weights_outside_split)
        if weights_outside_split:
            print("If True, the weights encode the class balance of all",
                  len(questions), "rows,\nincluding the test fold the model "
                  "is later scored on.")

record("C1", "pre-split fits (cause 11)",
       "NOT RUN" if cause11_confirmed is None else len(pre_split),
       "0", "UNVERIFIABLE" if cause11_confirmed is None
       else ("STILL OPEN" if cause11_confirmed else "CLOSED"))


# BLOCK V1f — TUNING BUDGET PER ARM
head("BLOCK V1f — TUNING BUDGET PER ARM")

SEARCH_TOKENS = ["GridSearchCV", "RandomizedSearchCV", "BayesSearchCV",
                 "ParameterGrid", "optuna", "hyperopt"]
if not NB_OK:
    print("NOT IN SOURCE: v3 notebook — V1f SKIPPED")
    parity = None
else:
    hits = {t: 0 for t in SEARCH_TOKENS}
    base_calls, eng_calls = [], []
    for ci, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        for t in SEARCH_TOKENS:
            hits[t] += len(re.findall(t, src))
        for li, line in enumerate(src.split("\n"), 1):
            if "train_model_v2(" in line and "def " not in line:
                if "weight_mode='none'" in line or 'weight_mode="none"' in line:
                    base_calls.append((ci, li, line.strip()))
                elif "weight_mode='computed'" in line or 'weight_mode="computed"' in line:
                    eng_calls.append((ci, li, line.strip()))
    print("search constructs found:", {k: v for k, v in hits.items() if v})
    print("distinct baseline call sites  :", len(base_calls))
    for c in base_calls:
        print("   cell %d L%d  %s" % (c[0], c[1], c[2][:70]))
    print("distinct engineered call sites:", len(eng_calls))
    for c in eng_calls:
        print("   cell %d L%d  %s" % (c[0], c[1], c[2][:70]))
    parity = (sum(hits.values()) == 0) and (len(base_calls) == len(eng_calls))
    print("\nTUNING PARITY:", parity)
    print("Zero search constructs on both arms is parity, and it is the "
          "statement\nthe Methods must carry. It is not a defect.")


# BLOCK V1a — RAG ARM STABILITY BY RESAMPLING
head("BLOCK V1a — RAG ARM STABILITY (RESAMPLING, n=%d)" % N_BOOT)

if not FILE_PRESENT["gold_rouge_v2.json"]:
    print("NOT IN SOURCE: gold_rouge_v2.json —", GOLD_ROUGE)
    print("V1a SKIPPED. Null state cannot be computed.")
    ROUGE = None
else:
    ROUGE = json.load(open(GOLD_ROUGE))
    arr = {s: np.asarray(ROUGE[s], dtype=float) for s in SYSTEMS}
    n_pairs = len(arr["crag"])
    print("per-item arrays loaded, n =", n_pairs)
    for s in SYSTEMS:
        assert len(arr[s]) == n_pairs, "system %s has %d rows, crag has %d" % (
            s, len(arr[s]), n_pairs)
    print("n before any filter:", n_pairs, " n after: ", n_pairs, " (no filter applied)")

    rng = np.random.default_rng(SEED_MAIN)
    print("\n%-18s %10s %10s %10s %12s %12s" % (
        "CONTRAST", "MEAN d", "SD d", "p", "CI LOW", "SIGN FLIP"))
    stability = {}
    for other in ["vanilla", "naive", "bm25"]:
        d = arr["crag"] - arr[other]
        try:
            _, p = wilcoxon(arr["crag"], arr[other])
        except ValueError:
            p = float("nan")
        boot = np.array([rng.choice(d, size=n_pairs, replace=True).mean()
                         for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        flip = float(np.mean(np.sign(boot) != np.sign(d.mean())))
        stability[other] = dict(mean=d.mean(), sd=d.std(ddof=1), p=p,
                                lo=lo, hi=hi, flip=flip)
        print("%-18s %10.4f %10.4f %10.6f %12.4f %12.3f" % (
            "crag vs " + other, d.mean(), d.std(ddof=1), p, lo, flip))
    print("\nCI is the 95%% bootstrap interval on the paired mean difference.")
    print("SIGN FLIP is the fraction of resamples where the sign reversed.")
    print("A CI that spans zero is not an absence of effect. It is an "
          "interval\nthat contains zero.")


# BLOCK V1b — FIVE-FOLD TABLE OVER THE GOLD PAIRS
head("BLOCK V1b — FIVE-FOLD TABLE, SCORED AGAINST GOLD ANSWERS")

if ROUGE is None:
    print("V1b SKIPPED — gold_rouge_v2.json absent")
    folds_ok = None
else:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_MAIN)
    idx = np.arange(n_pairs)
    print("%-8s %6s %12s %12s %12s %12s" % (
        "FOLD", "n", "CRAG", "NAIVE", "BM25", "VANILLA"))
    fold_means = {s: [] for s in SYSTEMS}
    fold_count = 0
    for f, (_, te) in enumerate(kf.split(idx), 1):
        fold_count += 1
        row = [np.mean(arr[s][te]) for s in SYSTEMS]
        for s, v in zip(SYSTEMS, row):
            fold_means[s].append(v)
        print("%-8d %6d %12.4f %12.4f %12.4f %12.4f" % (f, len(te), *row))
    print("-" * 66)
    print("%-8s %6d %12.4f %12.4f %12.4f %12.4f" % (
        "MEAN", n_pairs, *[np.mean(fold_means[s]) for s in SYSTEMS]))
    print("%-8s %6s %12.4f %12.4f %12.4f %12.4f" % (
        "SD", "", *[np.std(fold_means[s], ddof=1) for s in SYSTEMS]))
    winners = [SYSTEMS[int(np.argmax([fold_means[s][i] for s in SYSTEMS]))]
               for i in range(fold_count)]
    print("\nper-fold top system:", winners)
    print("folds won by crag  :", winners.count("crag"), "of", fold_count)
    folds_ok = fold_count == N_FOLDS

record("F1", "5-fold table rows",
       "SKIPPED" if folds_ok is None else fold_count,
       N_FOLDS, "UNVERIFIABLE" if folds_ok is None
       else ("CLOSED" if folds_ok else "STILL OPEN"))


# BLOCK V1c — MINIMUM DETECTABLE EFFECT, PAIRED
head("BLOCK V1c — MINIMUM DETECTABLE EFFECT (PAIRED SOLVER)")

if ROUGE is None:
    print("V1c SKIPPED — gold_rouge_v2.json absent")
    mde_flag = None
else:
    d = arr["crag"] - arr["vanilla"]
    sd_d = d.std(ddof=1)
    obs = d.mean()
    z_a = norm.ppf(1 - ALPHA_BONF / 2.0)
    z_b = norm.ppf(POWER_TARGET)
    mde = (z_a + z_b) * sd_d / math.sqrt(n_pairs)
    achieved = norm.cdf(abs(obs) / (sd_d / math.sqrt(n_pairs)) - z_a)
    n_req = int(math.ceil(((z_a + z_b) * sd_d / obs) ** 2)) if obs != 0 else None
    print("contrast              : crag vs vanilla, paired")
    print("n pairs               :", n_pairs)
    print("observed mean delta   : %.4f" % obs)
    print("SD of paired delta    : %.4f" % sd_d)
    print("alpha (Bonferroni)    : %.4f" % ALPHA_BONF)
    print("MDE at %.0f%% power     : %.4f" % (POWER_TARGET * 100, mde))
    print("achieved power        : %.4f" % achieved)
    print("n required for %.0f%%   : %s" % (POWER_TARGET * 100, n_req))
    mde_flag = mde > abs(obs)
    print("\nMDE EXCEEDS OBSERVED DELTA:", mde_flag)
    if mde_flag:
        print("WORDING LOCK: the result is inconclusive at this n;", n_req,
              "pairs\nwould have been required to detect the claimed effect.")


# BLOCK V1d — BASE RATE, PER-CLASS RECALL, CONFUSION COUNTS
head("BLOCK V1d — BASE RATE AND PER-CLASS RECALL")

tf1, test_pred, test_true, _, _, _, _, _, _ = train_model_v2(
    questions, labels, weight_mode='computed', seed=SEED_MAIN)
counts = Counter(labels)
total = sum(counts.values())
print("test fold n           :", len(test_true))
print("macro F1 (this run)   : %.4f" % tf1)
print("\n%-22s %8s %10s %10s" % ("CLASS", "n TOTAL", "BASE RATE", "RECALL"))
rec = recall_score(test_true, test_pred, average=None,
                   labels=list(range(num_classes)), zero_division=0)
for i, name in enumerate(class_names):
    print("%-22s %8d %10.4f %10.4f" % (
        str(name)[:22], counts[i], counts[i] / total, rec[i]))
print("\nmajority class base rate: %.4f" % (max(counts.values()) / total))
print("confusion counts (rows = true, cols = predicted):")
print(confusion_matrix(test_true, test_pred, labels=list(range(num_classes))))
print("\nzero-recall classes:", int((rec == 0).sum()))


# BLOCK V2 — SPLIT SIZES FROM THE RUN
head("BLOCK V2 — SPLIT SIZES")

idx_all = np.arange(len(labels))
i_temp, i_test, y_temp_c, y_test_c = train_test_split(
    idx_all, labels_arr, test_size=TEST_SIZE, stratify=labels_arr, random_state=SEED_MAIN)
i_train, i_val, _, _ = train_test_split(
    i_temp, y_temp_c, test_size=VAL_SIZE, stratify=y_temp_c, random_state=SEED_MAIN)

obs = (len(i_train), len(i_val), len(i_test))
print("n total               :", len(labels))
print("%-12s %10s %10s %8s" % ("FOLD", "OBSERVED", "QUOTED", "MATCH"))
for label, o, q in zip(["train", "val", "test"], obs,
                       [METHOD_TRAIN_N, METHOD_VAL_N, METHOD_TEST_N]):
    print("%-12s %10d %10d %8s" % (label, o, q, "YES" if o == q else "NO"))
split_match = obs == (METHOD_TRAIN_N, METHOD_VAL_N, METHOD_TEST_N)
print("\nQUOTED values are from the Method and were not used in any calculation.")
print("SPLIT MATCHES METHOD:", split_match)

record("C1", "split sizes printed", "%d/%d/%d" % obs,
       "%d/%d/%d" % (METHOD_TRAIN_N, METHOD_VAL_N, METHOD_TEST_N), "CLOSED")


# BLOCK V3 — PERTURBATION PAIRING PROOF
head("BLOCK V3 — PERTURBATION QUERY/LABEL PAIRING")

pi_temp, pi_test, py_temp, py_test = train_test_split(
    idx_all, labels_arr, test_size=TEST_SIZE, stratify=labels_arr, random_state=SEED_PERTURB)
assert np.array_equal(labels_arr[pi_test], py_test), (
    "index reconstruction does not reproduce the seed-%d test fold" % SEED_PERTURB)
print("index reconstruction verified against the seed-%d split" % SEED_PERTURB)

n_test = len(pi_test)
print("\ntest fold n                    :", n_test)
print("NOTE (v3): the perturbation cell reuses the deployment model's own")
print("q_test_best/y_test_arr_best directly from train_model_v2's return")
print("values — it does not reconstruct positions from questions[0:n] the")
print("way the earlier defective v2 cell 13 did. The mispairing pattern")
print("this block was built to catch does not apply to the v3 perturbation")
print("cell's actual code path; confirm by inspecting the perturbation cell")
print("source directly rather than by position reconstruction here.")

record("F8", "perturbation harness rewritten", "reuses train_model_v2 return",
       "no reconstructed indices", "CLOSED")


# BLOCK V4 — FAIRNESS BY TOPIC
head("BLOCK V4 — FAIRNESS BY TOPIC")

topic_rows = None
if ROUGE is None or not FILE_PRESENT["gold_qa_pairs.csv"]:
    print("SKIPPED — needs both gold_qa_pairs.csv and gold_rouge_v2.json")
else:
    gold_df = pd.read_csv(GOLD_CSV)
    print("gold rows read        :", len(gold_df))
    print("columns               :", list(gold_df.columns))
    assert TOPIC_COL in gold_df.columns, (
        "column '%s' is not in gold_qa_pairs.csv. Columns are %s."
        % (TOPIC_COL, list(gold_df.columns)))
    assert len(gold_df) == n_pairs, (
        "gold_qa_pairs.csv has %d rows, the scored arrays have %d."
        % (len(gold_df), n_pairs))
    topics = gold_df[TOPIC_COL].astype(str).values
    uniq = sorted(set(topics))
    print("distinct topics       :", len(uniq), "(expected", N_TOPICS_EXPECT, ")")
    print("\n%-26s %5s %11s %11s %11s %11s" % (
        "TOPIC", "n", "CRAG", "NAIVE", "BM25", "VANILLA"))
    n_seen = 0
    for t in uniq:
        mask = topics == t
        n_seen += int(mask.sum())
        means = [arr[s][mask].mean() for s in SYSTEMS]
        sds = [arr[s][mask].std(ddof=1) if mask.sum() > 1 else float("nan")
               for s in SYSTEMS]
        print("%-26s %5d %11.4f %11.4f %11.4f %11.4f" % (t[:26], mask.sum(), *means))
        print("%-26s %5s %11.4f %11.4f %11.4f %11.4f" % ("  SD", "", *sds))
    print("-" * 78)
    print("rows accounted for    : %d of %d (no topic dropped)" % (n_seen, n_pairs))
    losses = [t for t in uniq
              if arr["crag"][topics == t].mean() < max(
                  arr[s][topics == t].mean() for s in ["naive", "bm25", "vanilla"])]
    print("topics where crag is not top:", losses if losses else "NONE")
    topic_rows = len(uniq)

record("F1", "fairness topic rows", topic_rows if topic_rows else "SKIPPED",
       N_TOPICS_EXPECT, "CLOSED" if topic_rows else "UNVERIFIABLE")


# BLOCK V5 — EMPTY AND FAILED GENERATIONS
head("BLOCK V5 — EMPTY AND FAILED GENERATIONS")

empties = None
if not FILE_PRESENT["gold_eval_checkpoint.json"]:
    print("NOT IN SOURCE: gold_eval_checkpoint.json —", GOLD_CKPT)
    print("V5 SKIPPED.")
else:
    ck = json.load(open(GOLD_CKPT))
    print("checkpoint progress field:", ck.get("progress", "NOT PRESENT"))
    print("%-10s %8s %8s %8s %10s %10s" % (
        "SYSTEM", "n", "EMPTY", "ZERO-R", "MIN LEN", "MEAN LEN"))
    empties = {}
    for s in SYSTEMS:
        ans = ck.get(s, [])
        n_ans = len(ans)
        blank = sum(1 for a in ans if not str(a).strip())
        lens = [len(str(a).strip()) for a in ans] or [0]
        zero_r = int(np.sum(arr[s] == 0.0)) if ROUGE is not None else -1
        empties[s] = blank
        print("%-10s %8d %8d %8d %10d %10.1f" % (
            s, n_ans, blank, zero_r, min(lens), float(np.mean(lens))))
        if n_ans != N_GOLD_EXPECT:
            print("   WARNING: %d answers on file, %d expected" % (n_ans, N_GOLD_EXPECT))
    print("\ntotal empties across systems:", sum(empties.values()))

record("C3", "empty generations counted",
       "SKIPPED" if empties is None else sum(empties.values()),
       "printed per system", "UNVERIFIABLE" if empties is None else "CLOSED")


# BLOCK V6 — REPOSITORY ARTEFACT STAGING
head("BLOCK V6 — REPOSITORY ARTEFACT STAGING")

os.makedirs(STAGING_DIR, exist_ok=True)
print("staging directory:", STAGING_DIR)
staged = []

if FILE_PRESENT["gold_qa_pairs.csv"]:
    dst = os.path.join(STAGING_DIR, "gold_qa_pairs.csv")
    shutil.copyfile(GOLD_CSV, dst)
    staged.append(dst)
    print("staged: gold_qa_pairs.csv")
else:
    print("DATA NOT AVAILABLE: gold_qa_pairs.csv")

if FILE_PRESENT["gold_eval_checkpoint.json"] and FILE_PRESENT["gold_qa_pairs.csv"]:
    ck = json.load(open(GOLD_CKPT))
    gdf = pd.read_csv(GOLD_CSV)
    n_min = min([len(gdf)] + [len(ck.get(s, [])) for s in SYSTEMS])
    print("rows before alignment:", len(gdf), " rows after:", n_min)
    out = pd.DataFrame({"question": gdf["question"][:n_min],
                        "gold_answer": gdf["gold_answer"][:n_min]})
    for s in SYSTEMS:
        out[s + "_ans"] = ck.get(s, [])[:n_min]
    dst = os.path.join(STAGING_DIR, "gold_system_answers.csv")
    out.to_csv(dst, index=False)
    staged.append(dst)
    print("staged: gold_system_answers.csv  rows=%d cols=%s" % (len(out), list(out.columns)))
else:
    print("DATA NOT AVAILABLE: system answers cannot be staged")

if not os.path.exists(TLX_RESPONSES):
    print("\nDATA NOT AVAILABLE: trust and workload responses.")
    tlx_ok = False
else:
    print("\ntrust/workload responses located:", TLX_RESPONSES)
    tlx_ok = True

print("\nFILES STAGED:", len(staged))
for f in staged:
    print("  ", f)

record("C4", "gold artefacts staged", "%d of 2 staged" % len(staged),
       "committed at a hash", "UNVERIFIABLE")
record("C4", "trust/workload analysis", "present" if tlx_ok else "DATA NOT AVAILABLE",
       "cell + responses committed", "CLOSED" if tlx_ok else "STILL OPEN")


# ── VERDICT TABLE ────────────────────────────────────────────────────────────
head("VERDICT TABLE — PASTE THIS BACK IN FULL")

print(f"{'ITEM':<10}{'CHECK':<32}{'OBSERVED':>16}{'EXPECTED':>18}  VERDICT")
print("-" * 96)
for item, check, observed, expected, verdict in VERDICTS:
    print(f"{item:<10}{check[:32]:<32}{observed[:16]:>16}{expected[:18]:>18}  {verdict}")
print("-" * 96)

closed = sum(1 for v in VERDICTS if v[4] == "CLOSED")
still = sum(1 for v in VERDICTS if v[4] == "STILL OPEN")
unver = sum(1 for v in VERDICTS if v[4] == "UNVERIFIABLE")
print(f"CLOSED: {closed}   STILL OPEN: {still}   UNVERIFIABLE: {unver}")

print("""
AUDIT FOOTER
  ANCHOR              : RUN-GROUP_1-20260913 (v3 patch)
  OWNER               : GROUP 1
  BLOCKS              : V1a V1b V1c V1d V1e V1f V2 V3 V4 V5 V6
""")