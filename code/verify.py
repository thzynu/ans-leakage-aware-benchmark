"""Integrity check of the repository. Run from the repository root:  python code\\verify.py

Checks: (1) syntax of every code file, (2) result files present with the expected number of entries, valid probabilities,
(3) no personal strings in text files, (4) no file above 90 MB, (5) analysis.py reproduces the key numbers of the paper.
Private strings (names, paths) that must not appear in the published repository are read from the file
private_strings.txt in the repository root (one per line). Keep that file out of git (it is listed in .gitignore).
"""
import glob, os, pickle, re, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIVATE = ["@gmail", "/content/drive", "kaggle/input/datasets"]
_pf = os.path.join(ROOT, "private_strings.txt")
if os.path.exists(_pf):
    PRIVATE += [l.strip() for l in open(_pf, encoding="utf-8", errors="ignore") if l.strip()]
ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(("PASS  " if cond else "FAIL  ") + label + (f"   [{detail}]" if detail else ""))


# ---- 1. syntax
print("== 1. Code files")
code_files = sorted(glob.glob(os.path.join(ROOT, "code", "*.py")))
check("code/ contains python files", len(code_files) >= 8, f"{len(code_files)} files")
for f in code_files:
    try:
        compile(open(f, encoding="utf-8").read(), f, "exec"); check("syntax " + os.path.basename(f), True)
    except SyntaxError as e:
        check("syntax " + os.path.basename(f), False, str(e))
for name in ["common.py", "models_extra.py", "wesad_benchmark.py", "case_validation.py", "xai_stability.py",
             "xai_results.py", "analysis.py", "preprocess_wesad.py", "pre_case.py", "ml_baseline.py"]:
    check("present code/" + name, os.path.exists(os.path.join(ROOT, "code", name)))

# ---- 2. result files
print("\n== 2. Result files")
R = os.path.join(ROOT, "results")
EXPECT = {"results.pkl": {m: 45 for m in ["hybrid", "ecg", "eda", "resp", "ecg+eda", "ecg+resp", "eda+resp", "cnn_only", "rnn_only",
                                          "transformer", "crossmodal", "hyb_noattn", "hyb_uniform", "hyb_allbi", "hyb_alluni"]},
          "results_ml.pkl": {"svm": 15, "rf": 45},
          "results_fusion.pkl": {"fus_sum": 45, "fus_gated": 45, "fus_xattn": 45},
          "results_case.pkl": {n: 90 for n in ["hybrid", "cnn_only", "rnn_only", "tr_hybrid", "tr_cnn_only", "tr_rnn_only"]}}
for f, exp in EXPECT.items():
    p = os.path.join(R, f)
    if not os.path.exists(p):
        check(f + " exists", False); continue
    o = pickle.load(open(p, "rb"))
    cnt = {}
    for k in o: cnt[k[0]] = cnt.get(k[0], 0) + 1
    check(f + " entries per model", cnt == exp, f"total {len(o)}, expected {sum(exp.values())}")
    bad = [k for k, v in o.items() if not (len(v["probs"]) == len(v["y"]) and np.allclose(v["probs"].sum(1), 1, atol=1e-3))]
    check(f + " probabilities valid", not bad, f"{len(bad)} invalid")
    subj = sorted({k[2] for k in o})
    want = list(range(101, 131)) if "case" in f else [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    check(f + " subjects", subj == want, f"{len(subj)} subjects")
p = os.path.join(R, "xai_stability.pkl")
if os.path.exists(p):
    o = pickle.load(open(p, "rb"))
    check("xai_stability.pkl has 45 entries (15 subjects x 3 seeds), format 'maps'", len(o) == 45 and "maps" in next(iter(o.values())),
          f"{len(o)} entries; file size {os.path.getsize(p) / 1e6:.1f} MB (about 3.7 MB for the full run)")
else:
    check("xai_stability.pkl exists", False)
p = os.path.join(R, "xai_results.pkl")
if os.path.exists(p):
    o = pickle.load(open(p, "rb"))
    check("xai_results.pkl has 15 subjects with rob/delins/ent", len(o) == 15 and all({"rob", "delins", "ent"} <= set(v) for v in o.values()))
else:
    check("xai_results.pkl exists", False)

# ---- 3. personal strings, 4. large files
print("\n== 3. Personal strings in text files")
hits = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in (".git", "__pycache__")]
    for n in fn:
        if os.path.splitext(n)[1].lower() in (".py", ".md", ".txt", ".ipynb", ".json", ".cfg", ".csv"):
            if n == "private_strings.txt" or os.path.abspath(os.path.join(dp, n)) == os.path.abspath(__file__): continue
            t = open(os.path.join(dp, n), encoding="utf-8", errors="ignore").read().lower()
            for s in PRIVATE:
                if s.lower() in t: hits.append((os.path.relpath(os.path.join(dp, n), ROOT), s))
print("(" + str(len(PRIVATE)) + " patterns checked; add your own to private_strings.txt)")
check("no private strings found", not hits, "; ".join(f"{a}: '{b}'" for a, b in hits[:8]))
print("\n== 4. Large files")
big = [(os.path.relpath(os.path.join(dp, n), ROOT), os.path.getsize(os.path.join(dp, n)) / 1e6)
       for dp, dn, fn in os.walk(ROOT) if ".git" not in dp for n in fn if os.path.getsize(os.path.join(dp, n)) > 90e6]
check("no file above 90 MB (GitHub limit 100 MB)", not big, str(big))

# ---- 5. reproduce key numbers
print("\n== 5. analysis.py reproduces the key numbers")
r = subprocess.run([sys.executable, os.path.join(ROOT, "code", "analysis.py"), "--results", R], capture_output=True, text=True)
out = re.sub(r"[ \t]+", " ", r.stdout)
check("analysis.py runs without error", r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1])
EXPECTED = ["hybrid F1 0.707", "cnn_only F1 0.720", "rnn_only F1 0.728", "transformer F1 0.682", "crossmodal F1 0.737",
            "svm F1 0.711", "rf F1 0.716", "ecg F1 0.359", "eda F1 0.674", "Friedman (7 main models): p = 0.318",
            "hybrid 0.488", "cnn_only 0.477", "tr_hybrid 0.400", "hybrid arg max 0.707 | tuned 0.851 | inflation +0.144",
            "del: IG 0.464 random 0.520", "same: 0.626", "diff: 0.138", "same > diff in 15 / 15 subjects"]
for e in EXPECTED: check("expected: " + e, e in out)
m = re.search(r"fus_gated F1 (0\.\d{3})", out)
print("INFO  fus_gated macro-F1 =", m.group(1) if m else "?", "(0.738 = first fusion run, 0.736 = repeated run)")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED: fix the FAIL lines above")
sys.exit(0 if ok else 1)
