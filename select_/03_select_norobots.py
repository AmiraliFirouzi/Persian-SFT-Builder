# ============================================================
# Inventory + Selection — Persian-NoRobots ( 9,500  train)
# self-contained
# ============================================================
import hashlib, json, random, re
from collections import Counter
from datasets import load_dataset, get_dataset_config_names

REPO = "ParsBench/Persian-NoRobots"   
assert "/" in REPO and "..." not in REPO, "REPO را پر کن"
print("LOADING:", REPO)

cfgs = get_dataset_config_names(REPO)
if len(cfgs) > 1:
    raise SystemExit(f"این دیتاست {len(cfgs)} config دارد: {cfgs} — لیست را بفرست")
raw = load_dataset(REPO)
print("splits:", {k: len(v) for k, v in raw.items()})
assert "train" in raw, "split train پیدا نشد"

if "test" in raw:
    print(f"[GUARD] test split: {len(raw['test']):,} rows — LOCKED، هرگز وارد training نمی‌شود")
ds = raw["train"]

# ---------- Phase A: peek ----------
peek = [ds[i] for i in range(3)]
print("\nFIELDS:", list(peek[0].keys()))
for i, ex in enumerate(peek):
    print(f"\n--- sample {i} ---")
    for k, v in ex.items():
        print(f"{k}: {str(v)[:220]}")

cat_key = next((k for k in peek[0] if k.lower() in {"category", "task_category", "cat"}), None)
print("\ncategory field:", cat_key or "NOT FOUND")

# ---------- Phase B: scan ----------
TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

roles, turns, cats = Counter(), Counter(), Counter()
seen, rows, dups, bad = set(), [], 0, 0
for ex in ds:
    m = ex.get("messages")
    if not isinstance(m, list) or len(m) < 2 or \
       not all(x.get("role") in {"system", "user", "assistant"} for x in m):
        bad += 1; continue
    roles.update(x["role"] for x in m)
    turns[len(m)] += 1
    cat = str(ex[cat_key]) if cat_key else "unknown"
    cats[cat] += 1
    L = sum(len(str(x.get("content", ""))) for x in m)
    h = hashlib.md5("§".join(norm(str(x.get("content", ""))) for x in m).encode()).hexdigest()
    if h in seen: dups += 1; continue
    seen.add(h)
    rows.append({"messages": m, "category": cat, "length_chars": L})

lens = sorted(r["length_chars"] for r in rows)
p = lambda q: lens[int(q * (len(lens) - 1))]
print("\n===== REPORT =====")
print(f"usable={len(rows):,}  exact_dups={dups}  bad_structure={bad}")
print("roles:", dict(roles))
print("turns:", dict(sorted(turns.items())))
print(f"len_chars p50/p90/p99/max: {p(.5)} / {p(.9)} / {p(.99)} / {lens[-1]}")
print(f"categories ({len(cats)}):")
for c, n in cats.most_common(): print(f"  {c}: {n:,}")

# ---------- Phase C: save train ----------
random.Random(42).shuffle(rows)
with open("norobots_selected.jsonl", "w", encoding="utf-8") as f:
    for n, r in enumerate(rows):
        f.write(json.dumps({
            "id": f"norobots-{n:05d}", "source": "Persian-NoRobots",
            "source_subset": r["category"], "task": r["category"].lower(),
            "language": "fa", "thinking": False,
            "synthetic_or_human": "synthetic_translated", "original_language": "en",
            "length_chars": r["length_chars"], "quality_score": None,
            "messages": r["messages"],
        }, ensure_ascii=False) + "\n")

print(f"\nSAVED: norobots_selected.jsonl  ({len(rows):,} rows)")
if abs(len(rows) - 9500) > 190:
    print(f"⚠️ انحراف از 9,500: {len(rows):,} — قبل از ادامه بفرست بررسی کنم")
else:
    print(f"انحراف از 9,500: {9500 - len(rows)} ردیف (dedup) — در merge با buffer جبران می‌شود")

# ---------- Phase D: review ----------
sys_ex = next((r for r in rows if r["messages"][0]["role"] == "system"), None)
mt_ex  = next((r for r in rows if len(r["messages"]) > 2), None)
for label, r in [("با system", sys_ex), ("multi-turn", mt_ex)]:
    if r:
        print(f"\n--- نمونه {label} [{r['category']}] ---")
        for m in r["messages"][:3]:
            print(f"{m['role']}: {str(m.get('content',''))[:140]}")