# ============================================================
# Selection — PersianSyntheticQA  →  synqa_selected_15k.jsonl
# ============================================================
import hashlib, json, random, re
from collections import Counter, defaultdict
from datasets import load_dataset, get_dataset_config_names

REPO, SEED, PER_DOMAIN = "ParsBench/PersianSyntheticQA", 42, 300

configs = sorted(get_dataset_config_names(REPO))
assert len(configs) == 50, f"تعداد دامنه‌ها {len(configs)} شد"

rng = random.Random(SEED)
TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

# ---------- Phase 1: global dedup ----------
pool, seen, dups, skipped = defaultdict(list), set(), 0, 0
for cfg in configs:
    d = load_dataset(REPO, cfg)
    for ex in d["train"]:
        msgs = ex["messages"]
        i = str(msgs[0].get("content", ""))
        o = str(msgs[-1].get("content", ""))
        if len(i) < 15 or len(o) < 30:
            skipped += 1; continue
        h = hashlib.md5((norm(i) + "§" + norm(o)).encode()).hexdigest()
        if h in seen: dups += 1; continue
        seen.add(h)
        pool[cfg].append(msgs)
    print(f"  ✓ {cfg}: unique={len(pool[cfg])}")

# ---------- Phase 2: sampling 300 ----------
final, shortfall = [], {}
for cfg in configs:
    rows = pool[cfg][:]
    rng.shuffle(rows)
    take = rows[:PER_DOMAIN]
    if len(take) < PER_DOMAIN: shortfall[cfg] = PER_DOMAIN - len(take)
    final.extend((cfg, m) for m in take)
rng.shuffle(final)   

# ---------- Phase 3: standard schema ----------
with open("synqa_selected_15k.jsonl", "w", encoding="utf-8") as f:
    for n, (cfg, msgs) in enumerate(final):
        L = sum(len(m["content"]) for m in msgs)
        f.write(json.dumps({
            "id": f"synqa-{n:05d}", "source": "PersianSyntheticQA",
            "source_subset": cfg, "language": "fa", "task": "qa",
            "thinking": False, "synthetic_or_human": "synthetic",
            "original_language": "fa", "length_chars": L, "quality_score": None,
            "messages": msgs,
        }, ensure_ascii=False) + "\n")

# ---------- Phase 4: report ----------
per = Counter(cfg for cfg, _ in final)
turns = Counter(len(m) for _, m in final)
lens = sorted(sum(len(m["content"]) for m in msgs) for _, msgs in final)
p = lambda q: lens[int(q * (len(lens) - 1))]
print(f"\nFINAL: {len(final):,} / 15,000   SHORTFALL: {shortfall or 'none'}")
print(f"raw dups removed: {dups:,} | skipped (too short): {skipped}")
print("turns in final:", dict(sorted(turns.items())))
print(f"len_chars p50/p90/p99/max: {p(.5)} / {p(.9)} / {p(.99)} / {lens[-1]}")
print(f"per-domain min/max: {min(per.values())}/{max(per.values())}")

# ---------- Phase 5: review ----------
import itertools
for cfg, msgs in itertools.islice((x for x in final if len(x[1]) == 2), 3):
    print(f"\n--- [{cfg}] ---\nU: {msgs[0]['content'][:180]}\nA: {msgs[1]['content'][:180]}")
mt = next(((c, m) for c, m in final if len(m) > 2), None)
if mt:
    print(f"\n--- multi-turn [{mt[0]}] ({len(mt[1])} msgs) ---")
    for m in mt[1][:4]: print(f"{m['role']}: {m['content'][:120]}")