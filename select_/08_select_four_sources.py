# ============================================================
# Selection — 4 دیتاست جدید → 4 فایل jsonl با schema استاندارد
# ============================================================
import hashlib, json, random, re
from collections import Counter, defaultdict
from datasets import load_dataset

SEED, rng = 42, random.Random(42)
TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()
def h_of(u, a): return hashlib.md5((norm(u)[:300]+"§"+norm(a)[:300]).encode()).hexdigest()

def save(fname, prefix, rows, source, task, orig_lang="fa", synth="synthetic"):
    rng.shuffle(rows)
    with open(fname, "w", encoding="utf-8") as f:
        for n, m in enumerate(rows):
            f.write(json.dumps({
                "id": f"{prefix}-{n:05d}", "source": source, "source_subset": task,
                "task": task, "language": "fa", "thinking": False,
                "synthetic_or_human": synth, "original_language": orig_lang,
                "length_chars": sum(len(x["content"]) for x in m),
                "quality_score": None, "messages": m,
            }, ensure_ascii=False) + "\n")
    lens = sorted(sum(len(x["content"]) for x in m) for m in rows)
    print(f"SAVED {fname}: {len(rows):,} | p50={lens[len(lens)//2]} max={lens[-1]}")

# ---------- 1) wiki-instruct: 800 از هر task_type، سقف 2 به‌ازای هر مقاله ----------
ds = load_dataset("Jamalianpour/persian-wikipedia-instruct")["train"]
per_tt, src_cnt, seen, rows = defaultdict(list), Counter(), set(), []
for ex in ds:
    u, a = str(ex["instruction"]).strip(), str(ex["output"]).strip()
    if len(u) < 15 or len(a) < 30: continue
    k = h_of(u, a)
    if k in seen: continue
    sid, tt = str(ex["source_id"]), str(ex["task_type"])
    if src_cnt[(sid, tt)] >= 2: continue          # سقف هر مقاله-تسک
    seen.add(k); src_cnt[(sid, tt)] += 1
    per_tt[tt].append([{"role":"user","content":u},{"role":"assistant","content":a}])
for tt, lst in per_tt.items():
    rng.shuffle(lst); rows.extend(lst[:800])
save("wiki_instruct_selected.jsonl", "wikiin", rows, "PersianWikiInstruct", "knowledge_qa")
print("  per task_type:", {t: min(800, len(l)) for t, l in per_tt.items()})

# ---------- 2) maux-gpt: 2,500 با dedup ----------
ds = load_dataset("xmanii/maux-gpt-sft-20k")["train"]
seen, rows = set(), []
for ex in ds:
    m = ex["conversations"]
    msgs = [{"role": x["role"], "content": str(x["content"])} for x in m]
    u = next((x["content"] for x in msgs if x["role"]=="user"), "")
    a = next((x["content"] for x in reversed(msgs) if x["role"]=="assistant"), "")
    if len(u) < 15 or len(a) < 30: continue
    k = h_of(u, a)
    if k in seen: continue
    seen.add(k); rows.append(msgs)
save("mauxgpt_selected.jsonl", "mauxgpt", rows[:2500], "MauxGPT-SFT", "conversation",
     orig_lang="en", synth="synthetic_translated")

# ---------- 3) JumpShift: 800 با توازن category ----------
ds = load_dataset("jumplander/JumpShift")["train"]
per_cat, seen = defaultdict(list), set()
for ex in ds:
    u = str(ex["user_query"]).strip()
    a = str(ex["agent_response"]).strip()
    code = str(ex.get("code_example") or "").strip()
    if code:
        lang = str(ex.get("language") or "")
        a = f"{a}\n\n```{lang}\n{code}\n```" if lang else f"{a}\n\n```\n{code}\n```"
    if len(u) < 15 or len(a) < 30: continue
    k = h_of(u, a)
    if k in seen: continue
    seen.add(k)
    per_cat[str(ex["category"])].append(
        [{"role":"user","content":u},{"role":"assistant","content":a}])
# توازن: هر category سقف 150، سپس از باقی‌مانده پر می‌کنیم
rows, picked = [], Counter()
cats = sorted(per_cat, key=lambda c: -len(per_cat[c]))
while len(rows) < 800:
    progressed = False
    for c in cats:
        if picked[c] < len(per_cat[c]) and picked[c] < 150:
            rows.append(per_cat[c][picked[c]]); picked[c] += 1; progressed = True
            if len(rows) >= 800: break
    if not progressed: break
save("jumpshift_selected.jsonl", "jshift", rows, "JumpShift-FA", "code_fa")

# ---------- 4) Persian-Math: کل 1,000 ----------
ds = load_dataset("xmanii/Persian-Math-SFT")["train"]
seen, rows = set(), []
for ex in ds:
    m = ex["messages"]
    u = next((x["content"] for x in m if x["role"]=="user"), "")
    a = next((x["content"] for x in reversed(m) if x["role"]=="assistant"), "")
    if len(u) < 10 or len(a) < 30: continue
    k = h_of(u, a)
    if k in seen: continue
    seen.add(k)
    rows.append([{"role":"user","content":u},{"role":"assistant","content":a}])
save("pmath_selected.jsonl", "pmath", rows, "PersianMathSFT", "math_fa")