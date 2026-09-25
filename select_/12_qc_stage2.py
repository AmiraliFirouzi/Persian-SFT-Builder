# ============================================================
# MERGE Stage 2 — طول (Qwen tokenizer) + contamination + near-dup
# همه به‌صورت qc_flags — هیچ حذفی
# ============================================================
import hashlib, json, re
from collections import Counter
from transformers import AutoTokenizer
from datasets import load_dataset

rows = [json.loads(l) for l in open("master_stage1.jsonl", encoding="utf-8")]
print(f"loaded: {len(rows):,}")

TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

# ============ 1) Length audit با tokenizer واقعی Qwen ============
TOK = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(TOK)
LIMIT = 8192

def ntok(msgs):
    return len(tok.apply_chat_template(msgs, tokenize=True))

over, none_flagged = 0, 0
for r in rows:
    t = ntok(r["messages"])
    if t > LIMIT:
        r["qc_flags"].append("len_over_8k_tokens"); over += 1
print(f"[length] over {LIMIT} tokens: {over:,}")
lens = sorted(ntok(r["messages"]) for r in rows[:20000])   # نمونه 20K برای توزیع
p = lambda q: lens[int(q*(len(lens)-1))]
print(f"[length] token p50/p90/p99: {p(.5)} / {p(.9)} / {p(.99)}")

# ============ 2) Contamination — PerMMLU + PersianMedQA + ParsiNLU-MC ============
EVALS = {
    "permmlu":       "MCINext/permmlu",        # ← آدرس دقیق را بگذار
    "persianmedqa":  "MohammadJRanjbar/PersianMedQA",   # ← آدرس دقیق را بگذار
    "parsinlu_mc":   "PartAI/ParsiNLU-multiple-choice",# ← آدرس دقیق/درست را بگذار
}
NGRAM, flags_hit = 8, Counter()
eval_grams = {}
for name, repo in EVALS.items():
    try:
        d = load_dataset(repo)["train" if "train" in load_dataset(repo) else "test"]
        qcol = next((c for c in d.column_names if c.lower() in
                     {"question","prompt","query","inputs","text"}), d.column_names[0])
        grams = set()
        for ex in d:
            words = norm(str(ex[qcol])).split()
            for i in range(max(0, len(words)-NGRAM+1)):
                grams.add(" ".join(words[i:i+NGRAM]))
        eval_grams[name] = grams
        print(f"[decon] {name}: {len(d):,} questions → {len(grams):,} {NGRAM}-grams")
    except Exception as e:
        print(f"[decon] {name}: FAILED — {str(e)[:100]} (آدرس را درست کن)")

for r in rows:
    words = norm(" ".join(str(x.get("content") or "") for x in r["messages"]
                          if x["role"] in {"user","system"})).split()
    ss = {" ".join(words[i:i+NGRAM]) for i in range(max(0, len(words)-NGRAM+1))}
    for name, grams in eval_grams.items():
        if ss & grams:
            r["qc_flags"].append(f"contamination_{name}"); flags_hit[name] += 1
print(f"[decon] flagged: {dict(flags_hit) or 'هیچ'}")

# ============ 3) Near-dup فقط same-question/different-answer ============
q_seen, conflict = {}, 0
for r in rows:
    u = next((x["content"] for x in r["messages"] if x["role"]=="user"), "")
    q = norm(u)[:400]
    if not q: continue
    k = hashlib.md5(q.encode()).hexdigest()
    if k in q_seen and q_seen[k] != r["source"]:
        r["qc_flags"].append("near_dup_conflict"); conflict += 1
    elif k not in q_seen:
        q_seen[k] = r["source"]
print(f"[near-dup] same-question cross-source conflicts: {conflict:,}")

# ============ ذخیره ============
with open("master_final.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fc = Counter(fl for r in rows for fl in r["qc_flags"])
clean = sum(1 for r in rows if not r["qc_flags"])
print(f"\n===== FINAL =====\ntotal: {len(rows):,} | clean (no flags): {clean:,}")
print("flags:", dict(fc))
print("saved: master_final.jsonl")