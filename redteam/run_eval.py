"""Red-team the guard and report robustness.

Reports three numbers, each with a 95% Wilson confidence interval:
  - initial bypass-rate : fraction of held-out MALICIOUS prompts the guard misses (FN rate)
  - false-positive rate : fraction of held-out BENIGN prompts the guard wrongly blocks
  - residual bypass-rate: fraction of attack SEEDS for which >=1 mutated variant evades
                          the guard (the catch -> mutate -> re-test loop)
Plus a per-attack-family bypass breakdown. Writes results/redteam.json.

Co-reporting the FPR matters: you can trivially drive bypass-rate to 0 by blocking
everything, so a low bypass-rate is only meaningful next to a low FPR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from detector.predict import Detector  # noqa: E402
from redteam import mutators  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", action="store_true",
                    help="wrap the detector in the input-normalization defense")
    args = ap.parse_args()

    det = Detector()
    if args.defense:
        from detector.defenses import NormalizingDetector
        det = NormalizingDetector(det)
    out_path = config.RESULTS_DIR / ("redteam_defended.json" if args.defense else "redteam.json")
    test = pd.read_parquet(config.TEST_PARQUET)
    pos = test[test.label == 1].text.tolist()  # must be flagged
    neg = test[test.label == 0].text.tolist()  # must NOT be flagged

    pos_pred = det.predict_batch(pos)
    neg_pred = det.predict_batch(neg)
    n_bypass = sum(1 for lab, _ in pos_pred if lab == "benign")
    n_fp = sum(1 for lab, _ in neg_pred if lab == "malicious")
    bypass_rate, b_lo, b_hi = config.wilson_ci(n_bypass, len(pos))
    fpr, f_lo, f_hi = config.wilson_ci(n_fp, len(neg))

    # catch -> mutate -> re-test on the seed adversarial set
    per_family = {name: {"tried": 0, "bypassed": 0} for name in mutators.MUTATORS}
    seed_results, seed_bypassed = [], 0
    for seed in mutators.SEEDS:
        variants = mutators.all_variants(seed)
        row = {"seed": seed, "original_verdict": det.predict(seed)[0], "bypassed_by": []}
        bypassed = False
        for name, prompt in variants.items():
            if name == "original":
                continue
            label, _ = det.predict(prompt)
            per_family[name]["tried"] += 1
            if label == "benign":
                per_family[name]["bypassed"] += 1
                row["bypassed_by"].append(name)
                bypassed = True
        seed_bypassed += int(bypassed)
        seed_results.append(row)
    resid_rate, r_lo, r_hi = config.wilson_ci(seed_bypassed, len(mutators.SEEDS))

    out = {
        "n_pos": len(pos), "n_neg": len(neg), "seed_count": len(mutators.SEEDS),
        "initial_bypass_rate": {"value": bypass_rate, "ci95": [b_lo, b_hi], "count": n_bypass},
        "false_positive_rate": {"value": fpr, "ci95": [f_lo, f_hi], "count": n_fp},
        "residual_bypass_rate_after_mutation": {"value": resid_rate, "ci95": [r_lo, r_hi],
                                                "count": seed_bypassed},
        "per_family": {k: {**v, "bypass_rate": (v["bypassed"] / v["tried"] if v["tried"] else 0.0)}
                       for k, v in per_family.items()},
        "seed_results": seed_results,
    }
    out["defense"] = "input_normalization+keyword_ensemble" if args.defense else "none"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"=== Red-team report (defense: {out['defense']}) ===")
    print(f"Held-out positives={len(pos)}  negatives={len(neg)}")
    print(f"Initial bypass-rate (false negatives): {bypass_rate:6.1%}  95%CI[{b_lo:.1%}, {b_hi:.1%}]")
    print(f"False-positive rate (benign blocked):  {fpr:6.1%}  95%CI[{f_lo:.1%}, {f_hi:.1%}]")
    print(f"Residual bypass after mutation:        {resid_rate:6.1%}  95%CI[{r_lo:.1%}, {r_hi:.1%}]")
    print("\nPer-family bypass rate (over caught seeds):")
    for name, v in sorted(out["per_family"].items(), key=lambda kv: -kv[1]["bypass_rate"]):
        print(f"  {name:>18}: {v['bypass_rate']:4.0%}  ({v['bypassed']}/{v['tried']})")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
