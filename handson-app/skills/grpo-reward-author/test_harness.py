"""Run reward-function test cases and print a JSON report.

  python3 test_harness.py custom_reward.py tests.json

tests.json: [{"response": "...", "ground_truth": "...", "expected": 1.0, "why": "..."}, ...]
Exit code 0 if every case matches `expected` (within 1e-6) and every call returns a float in
[0, 1] without raising; 1 otherwise. Standard library only.
"""
import importlib.util
import json
import sys
import time


def load(path):
    spec = importlib.util.spec_from_file_location("custom_reward", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "compute_score"):
        raise SystemExit(json.dumps({"error": "compute_score not defined"}))
    return mod.compute_score


def main():
    fn = load(sys.argv[1])
    cases = json.load(open(sys.argv[2]))
    results, failed = [], 0
    for i, c in enumerate(cases):
        t0 = time.perf_counter()
        try:
            got = fn("custom", c["response"], c["ground_truth"], extra_info=None)
            err = None
        except Exception as e:  # the function must never raise
            got, err = None, f"{type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        ok = (err is None and isinstance(got, (int, float)) and 0.0 <= float(got) <= 1.0
              and abs(float(got) - float(c["expected"])) < 1e-6)
        failed += not ok
        results.append({"i": i, "ok": ok, "expected": c["expected"], "got": got, "error": err,
                        "ms": round(ms, 3), "why": c.get("why", ""), "ground_truth": c["ground_truth"],
                        "response_tail": str(c["response"])[-120:]})
    report = {"passed": len(cases) - failed, "failed": failed, "total": len(cases), "results": results}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
