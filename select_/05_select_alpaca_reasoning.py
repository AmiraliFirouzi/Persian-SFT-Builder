# ============================================================
# Selection — Persian Alpaca Reasoning ( 2,199)
# ============================================================
import hashlib, json, random, re
from collections import Counter
from datasets import load_dataset

REPO = "hosseinhimself/persian-alpaca-reasoning-v1"   
assert "/" in REPO and "..." not in REPO, "REPO را پر کن"
raw = load_dataset(REPO)
print("splits:", {k: len(v) for k, v in raw.items()})
ds = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]

# ---------- peek ----------
print("\nFIELDS:", list(ds[0].keys()))
for i in range(2):
    print(f"--- {i} ---")
    for k, v in ds[i].items(): print(f"{k}: {str(v)[:180]}")

TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

TERM = set(".!?؟…؛۔\n\"')»")
rows, seen, stats = [], set(), Counter()
for ex in ds:
    stats["total"] += 1
    ins  = str(ex.get("instruction", "") or "").strip()
    rea  = str(ex.get("reasoning", "") or "").strip()
    out  = str(ex.get("output", "") or "").strip()
    if len(ins) < 10: stats["bad_instruction"] += 1; continue
    if len(out) < 5:  stats["bad_output"] += 1; continue
    if len(rea) < 50: stats["reasoning_too_short"] += 1; continue
 
     
    tail = rea.rstrip()
    while tail and tail[-1] in "*_`~":   
        tail = tail[:-1]
    if not tail or tail[-1] not in TERM:
        stats["reasoning_incomplete"] += 1
        if stats["reasoning_incomplete"] <= 2:
            print(f"\n[نمونه reasoning ناقص] ...{rea[-120:]}")
        continue
    h = hashlib.md5((norm(ins) + "§" + norm(out)).encode()).hexdigest()
    if h in seen: stats["dups"] += 1; continue
    seen.add(h)
    stats["ok"] += 1
    rows.append({"ins": ins, "rea": rea, "out": out,
                 "length_chars": len(ins) + len(rea) + len(out)})


inc = sum(1 for r in rows if norm(r["out"][:60]) not in norm(r["rea"]))
print("\n===== REPORT =====")
print(dict(stats))
print(f"kept={len(rows)}  | output در reasoning نیامده: {inc} ({100*inc/max(len(rows),1):.1f}%) — اطلاعاتی")

random.Random(42).shuffle(rows)
with open("alpaca_reasoning_selected.jsonl", "w", encoding="utf-8") as f:
    for n, r in enumerate(rows):
        f.write(json.dumps({
            "id": f"alprea-{n:05d}", "source": "Persian-Alpaca-Reasoning",
            "source_subset": "reasoning", "task": "reasoning", "language": "fa",
            "thinking": True, "synthetic_or_human": "synthetic_translated",
            "original_language": "en", "length_chars": r["length_chars"],
            "quality_score": None, "reasoning": r["rea"],
            "messages": [{"role": "user", "content": r["ins"]},
                         {"role": "assistant", "content": r["out"]}],
        }, ensure_ascii=False) + "\n")
print(f"\nSAVED: alpaca_reasoning_selected.jsonl ({len(rows):,} rows)")
if abs(len(rows) - 2199) > 220:
    print(f"⚠️ kept از 2,199 دور است — خروجی را بفرست قبل از ادامه")

for r in rows[:3]:
    print(f"\nU: {r['ins'][:140]}\nTHINK: {r['rea'][:180]}\nA: {r['out'][:140]}")