# test_pipeline.py  (put it in the same folder as ai_module_code.py)

from ai_module_code import run_analysis_pipeline

print("Running pipeline...")
results = run_analysis_pipeline()

if not results:
    print("FAIL: pipeline returned empty list")
else:
    print(f"PASS: got {len(results)} tickers\n")
    for r in results:
        print(r)