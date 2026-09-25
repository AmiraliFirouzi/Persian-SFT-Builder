# ============================================================
# Selection — Aya (Persian) → aya_selected.jsonl
# ============================================================
import hashlib, json, random, re
from collections import Counter
from datasets import load_dataset

REPO = "CohereForAI/aya_dataset"
LANG = "Iranian Persian"
raw = load_dataset(REPO)
ds = raw["train"]                      

TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

rows, seen, stats = [], set(), Counter()
for ex in ds:
    if ex["language"] != LANG: continue
    stats["total"] += 1
    i, o = str(ex["inputs"]).strip(), str(ex["targets"]).strip()
    if len(i) < 15 or len(o) < 10:
        stats["too_short"] += 1; continue
    stats["annotation_type:" + str(ex["annotation_type"])] += 1
    h = hashlib.md5((norm(i) + "§" + norm(o)).encode()).hexdigest()
    if h in seen: stats["dups"] += 1; continue
    seen.add(h)
    rows.append({"inputs": i, "targets": o,
                 "ann": ex["annotation_type"],
                 "length_chars": len(i) + len(o)})

lens = sorted(r["length_chars"] for r in rows)
p = lambda q: lens[int(q * (len(lens) - 1))]
print("===== REPORT =====")
print(f"total_fa={stats['total']}  kept={len(rows)}  too_short={stats['too_short']}  dups={stats['dups']}")
print("annotation types:", {k: v for k, v in stats.items() if k.startswith('annotation_type')})
print(f"len_chars p50/p90/p99/max: {p(.5)} / {p(.9)} / {p(.99)} / {lens[-1]}")

random.Random(42).shuffle(rows)
with open("aya_selected.jsonl", "w", encoding="utf-8") as f:
    for n, r in enumerate(rows):
        f.write(json.dumps({
            "id": f"aya-{n:05d}", "source": "Aya",
            "source_subset": "pes", "task": "instruction", "language": "fa",
            "thinking": False, "synthetic_or_human": "human",
            "original_language": "fa", "length_chars": r["length_chars"],
            "quality_score": None,
            "messages": [{"role": "user", "content": r["inputs"]},
                         {"role": "assistant", "content": r["targets"]}],
        }, ensure_ascii=False) + "\n")
print(f"\nSAVED: aya_selected.jsonl ({len(rows):,} rows)")

# review
random.Random(7).shuffle(rows)
for r in rows[:6]:
    print(f"\nU: {r['inputs'][:160]}\nA: {r['targets'][:160]}")