import json, os
import pandas as pd
from datasets import Dataset

rows = [json.loads(l) for l in open("master_final.jsonl", encoding="utf-8")]
fa = [r for r in rows if r["source"] != "Replay-smoltalk2"]
KEEP = ["id","source","source_subset","task","language","thinking",
        "synthetic_or_human","original_language","length_chars",
        "length_tokens","quality_score","qc_flags","reasoning","messages"]

os.makedirs("publish_v2", exist_ok=True)
for name, data in [("fa", fa), ("fa_replay", rows)]:
    df = pd.DataFrame([{k: r.get(k) for k in KEEP if k in r} for r in data])
    # ستون‌های لیستی برای parquet باید رشته/لیست ساده باشند — json.dumps برای qc_flags
    df["qc_flags"] = df["qc_flags"].apply(lambda x: x or [])
    Dataset.from_pandas(df, preserve_index=False).to_parquet(f"publish_v2/{name}-00000-of-00001.parquet")
print(os.listdir("publish_v2"))