# ============================================================
# Selection — persian-gk (با پاک‌سازی system) + JumpShift top-up
# ============================================================
import hashlib, json, random, re
from collections import Counter, defaultdict
from datasets import load_dataset

SEED, rng = 42, random.Random(42)
TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()
def h_of(u, a): return hashlib.md5((norm(u)[:300]+"§"+norm(a)[:300]).encode()).hexdigest()

# fingerprint های فایل‌های خودمان برای overlap
ours = set()
for fp in ["synqa_selected_15k.jsonl", "mauxgpt_selected.jsonl"]:
    for l in open(fp, encoding="utf-8"):
        m = json.loads(l)["messages"]
        u = next((x["content"] for x in m if x["role"]=="user"), "")
        a = next((x["content"] for x in reversed(m) if x["role"]=="assistant"), "")
        ours.add(h_of(u, a))

# ---------- 1) persian-gk ----------
ds = load_dataset("mshojaei77/persian-gk")["train"]
seen, rows, sys_fixed = set(), [], 0
for ex in ds:
    msgs = [{"role": x["role"], "content": str(x["content"])} for x in ex["messages"]]
    # پاک‌سازی: system انگلیسی → حذف (بدنه فارسی خودگویاست)
    if msgs[0]["role"] == "system" and not re.search(r"[\u0600-\u06FF]", msgs[0]["content"]):
        msgs = msgs[1:]; sys_fixed += 1
    u = next((m["content"] for m in msgs if m["role"]=="user"), "")
    a = next((m["content"] for m in reversed(msgs) if m["role"]=="assistant"), "")
    if len(u) < 15 or len(a) < 30: continue
    k = h_of(u, a)
    if k in seen or k in ours: continue
    seen.add(k); rows.append(msgs)


rng.shuffle(rows)
with open("gk_selected.jsonl", "w", encoding="utf-8") as f:
    for n, m in enumerate(rows[:2000]):
        f.write(json.dumps({
            "id": f"gk-{n:05d}", "source": "PersianGK", "source_subset": "multiturn_gk",
            "task": "conversation", "language": "fa", "thinking": False,
            "synthetic_or_human": "synthetic",
            "original_language": "fa",
            "length_chars": sum(len(x["content"]) for x in m),
            "quality_score": None, "messages": m,
        }, ensure_ascii=False) + "\n")
print(f"gk: kept={len(rows):,} saved=2,000 | system-english removed: {sys_fixed:,}")
mt = sum(1 for m in rows[:2000] if len(m) > 3)
print(f"multi-turn in saved: {mt}")

# ---------- 2) JumpShift top-up: +200 بدون overlap با 800 قبلی ----------
ds = load_dataset("jumplander/JumpShift")["train"]
have = set()
for l in open("jumpshift_selected.jsonl", encoding="utf-8"):
    m = json.loads(l)["messages"]
    u = next((x["content"] for x in m if x["role"]=="user"), "")
    a = next((x["content"] for x in reversed(m) if x["role"]=="assistant"), "")
    have.add(h_of(u, a))
per_cat, seen = defaultdict(list), set()
for ex in ds:
    u = str(ex["user_query"]).strip()
    a0 = str(ex["agent_response"]).strip()
    code = str(ex.get("code_example") or "").strip()
    lang = str(ex.get("language") or "")
    a = (a0 + (f"\n\n```{lang}\n{code}\n```" if lang else f"\n\n```\n{code}\n```")) if code else a0
    if len(u) < 15 or len(a) < 30: continue
    k = h_of(u, a)
    if k in seen or k in have: continue
    seen.add(k)
    per_cat[str(ex["category"])].append(
        [{"role":"user","content":u},{"role":"assistant","content":a}])
rows, picked, cats = [], Counter(), sorted(per_cat, key=lambda c: -len(per_cat[c]))
while len(rows) < 200:
    progressed = False
    for c in cats:
        if picked[c] < len(per_cat[c]) and picked[c] < 40:
            rows.append(per_cat[c][picked[c]]); picked[c] += 1; progressed = True
            if len(rows) >= 200: break
    if not progressed: break
with open("jumpshift_extra_200.jsonl", "w", encoding="utf-8") as f:
    for n, m in enumerate(rows):
        f.write(json.dumps({
            "id": f"jshift-x-{n:05d}", "source": "JumpShift-FA", "source_subset": str(picked and ""),
            "task": "code_fa", "language": "fa", "thinking": False,
            "synthetic_or_human": "synthetic",
            "original_language": "fa",
            "length_chars": sum(len(x["content"]) for x in m),
            "quality_score": None, "messages": m,
        }, ensure_ascii=False) + "\n")
print(f"jumpshift top-up: {len(rows)} | per-category: {dict(picked)}")