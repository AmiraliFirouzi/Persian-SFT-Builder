import hashlib, re, json, random
from collections import Counter, defaultdict
from datasets import load_dataset

REPO, SEED, TARGET, BUFFER = "ParsiAI/FarsInstruct", 42, 20_000, 1.15  

QUOTAS = {  
    "persiannlp/parsinlu_translation_en_fa": 1000, "persiannlp/parsinlu_translation_fa_en": 1000,
    "pn_summary": 2500, "wiki_summary": 1500,
    "SajjadAyoubi/persian_qa": 3000, "parsinlu_reading_comprehension": 1800,
    "PNLPhub/FarsTail": 1200, "persiannlp/parsinlu_entailment": 500,
    "PNLPhub/C-ExaPPC": 2000, "PNLPhub/parsinlu-multiple-choice": 500,
    "persiannlp/parsinlu_query_paraphrasing": 800, "SLPL/syntran-fa": 800,
    "p3_xlwic": 500, "PNLPhub/Persian-News": 350, "PNLPhub/DigiMag": 350,
    "PNLPhub/snappfood-sentiment-analysis": 500, "persiannlp/parsinlu_sentiment": 400,
    "PNLPhub/digikala-sentiment-analysis": 250, "PNLPhub/Pars-ABSA": 250,
    "persian_ner": 400, "PNLPhub/PEYMA": 400,
}
TASK_OF = {
    "persiannlp/parsinlu_translation_en_fa": "translation", "persiannlp/parsinlu_translation_fa_en": "translation",
    "pn_summary": "summarization", "wiki_summary": "summarization",
    "SajjadAyoubi/persian_qa": "qa", "parsinlu_reading_comprehension": "reading_comprehension",
    "PNLPhub/FarsTail": "entailment", "persiannlp/parsinlu_entailment": "entailment",
    "PNLPhub/C-ExaPPC": "multiple_choice", "PNLPhub/parsinlu-multiple-choice": "multiple_choice",
    "persiannlp/parsinlu_query_paraphrasing": "paraphrasing", "SLPL/syntran-fa": "structured_generation",
    "p3_xlwic": "commonsense", "PNLPhub/Persian-News": "classification", "PNLPhub/DigiMag": "classification",
    "PNLPhub/snappfood-sentiment-analysis": "sentiment", "persiannlp/parsinlu_sentiment": "sentiment",
    "PNLPhub/digikala-sentiment-analysis": "sentiment", "PNLPhub/Pars-ABSA": "sentiment",
    "persian_ner": "ner", "PNLPhub/PEYMA": "ner",
}
assert sum(QUOTAS.values()) == TARGET and set(QUOTAS) == set(TASK_OF)

TEMP_CAP = {"persiannlp/parsinlu_translation_en_fa": 300, "persiannlp/parsinlu_translation_fa_en": 300}
rng = random.Random(SEED)

def to_text(v):
    if isinstance(v, list):
        v = v[0] if len(v) == 1 else " | ".join(str(x) for x in v)
    return str(v if v is not None else "").strip()

TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

# ---------- Phase 1: reservoir sampling per dataset ( train) ----------
ds = load_dataset(REPO, split="train")            
need = {d: int(q * BUFFER) for d, q in QUOTAS.items()}
K    = {d: int(n * 1.5) for d, n in need.items()}
reservoir, seen_n = defaultdict(list), Counter()

for ex in ds:
    d = ex["dataset"]
    if d not in need: continue
    seen_n[d] += 1
    row = (ex["template"], ex["inputs"], ex["outputs"])
    if len(reservoir[d]) < K[d]:
        reservoir[d].append(row)
    else:
        j = rng.randint(0, seen_n[d] - 1)
        if j < K[d]: reservoir[d][j] = row

# ---------- Phase 2: cap per-template ----------
pool = []
for d, rows in reservoir.items():
    rng.shuffle(rows)
    cap, tcnt, taken = TEMP_CAP.get(d, 400), Counter(), 0
    for t, i, o in rows:
        if taken >= need[d]: break
        if tcnt[t] >= cap: continue
        i, o = to_text(i), to_text(o)
        if len(i) < 10 or len(o) < 2: continue
        tcnt[t] += 1; taken += 1
        pool.append({"dataset": d, "template": t, "inputs": i, "outputs": o})

# ---------- Phase 3: exact dedup  ----------
seen, deduped = set(), []
for r in pool:
    h = hashlib.md5((norm(r["inputs"]) + "§" + norm(r["outputs"])).encode()).hexdigest()
    if h not in seen:
        seen.add(h); deduped.append(r)

# ---------- Phase 4: trim  ----------
by_d = defaultdict(list)
for r in deduped: by_d[r["dataset"]].append(r)
final, shortfall = [], {}
for d, q in QUOTAS.items():
    rows = by_d.get(d, []); rng.shuffle(rows)
    final.extend(rows[:q])
    if len(rows) < q: shortfall[d] = q - len(rows)

# ---------- Phase 5: save and report ----------
per_task = Counter(TASK_OF[r["dataset"]] for r in final)
lens = sorted(len(r["inputs"]) + len(r["outputs"]) for r in final)
p = lambda q: lens[int(q * (len(lens) - 1))]
print(f"FINAL: {len(final):,} / {TARGET:,}   candidates_after_dedup: {len(deduped):,}")
print("SHORTFALL:", shortfall if shortfall else "none")
print("per-task:", dict(per_task))
print(f"len_chars p50/p90/p99/max: {p(.5)} / {p(.9)} / {p(.99)} / {lens[-1]}")

with open("farsinstruct_selected_20k.jsonl", "w", encoding="utf-8") as f:
    for n, r in enumerate(final):
        f.write(json.dumps({
            "id": f"farsinstruct-{n:05d}", "source": "FarsInstruct",
            "source_subset": r["dataset"], "template": r["template"],
            "task": TASK_OF[r["dataset"]], "language": "fa", "thinking": False,
            "synthetic_or_human": "synthetic_translated", "original_language": "fa",
            "length_chars": len(r["inputs"]) + len(r["outputs"]), "quality_score": None,
            "messages": [{"role": "user", "content": r["inputs"]},
                         {"role": "assistant", "content": r["outputs"]}],
        }, ensure_ascii=False) + "\n")

# ---------- Phase 6: sample ----------
for d in ["p3_xlwic", "SLPL/syntran-fa", "PNLPhub/C-ExaPPC", "persiannlp/parsinlu_translation_fa_en"]:
    r = next(x for x in final if x["dataset"] == d)
    print(f"\n--- {d} ---\nIN : {r['inputs'][:200]}\nOUT: {r['outputs'][:200]}")
print("\nsaved: farsinstruct_selected_20k.jsonl")