# ============================================================
# Length audit v3 — صریح: قالب → رشته → encode
# ============================================================
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
rows = [json.loads(l) for l in open("master_final.jsonl", encoding="utf-8")]

def ntok(msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])

lens, over = [], 0
for r in rows:
    t = ntok(r["messages"])
    r["length_tokens"] = t
    lens.append(t)
    fl = r.setdefault("qc_flags", [])
    if t > 8192 and "len_over_8k_tokens" not in fl:
        fl.append("len_over_8k_tokens"); over += 1

lens.sort(); p = lambda q: lens[int(q*(len(lens)-1))]
print(f"tokens p50/p90/p99/max: {p(.5):,} / {p(.9):,} / {p(.99):,} / {lens[-1]:,}")
print(f"over 8192 tokens: {over:,}")

with open("master_final.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("updated.")