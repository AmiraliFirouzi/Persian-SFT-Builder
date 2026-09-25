# ============================================================
# MERGE Stage 1 (v2 — بدون حذف) → master_stage1.jsonl + qc_flags
# ============================================================
import hashlib, json, re
from collections import Counter

TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

FILES = [
    ("maux_persian_sft_25k_v7.jsonl",       "Maux-Persian-SFT"),
    ("farsinstruct_selected_20k.jsonl",     "FarsInstruct"),
    ("synqa_selected_15k.jsonl",            "PersianSyntheticQA"),
    ("norobots_selected.jsonl",             "Persian-NoRobots"),
    ("wiki_instruct_selected.jsonl",        "PersianWikiInstruct"),
    ("mauxgpt_selected.jsonl",              "MauxGPT-SFT"),
    ("aya_selected.jsonl",                  "Aya"),
    ("jumpshift_selected.jsonl",            "JumpShift-FA"),
    ("jumpshift_extra_200.jsonl",           "JumpShift-FA"),
    ("pmath_selected.jsonl",                "PersianMathSFT"),
    ("gk_selected.jsonl",                   "PersianGK"),
    ("alpaca_reasoning_selected.jsonl",     "Persian-Alpaca-Reasoning"),
    ("persian_thinking_selected.jsonl",     "Persian-Thinking"),
    ("replay_selected.jsonl",               "Replay-smoltalk2"),
]

def valid_msgs(m):
    return (isinstance(m, list) and len(m) >= 2 and
            all(isinstance(x, dict) and x.get("role") in
                {"system","user","assistant","tool"} and
                isinstance(x.get("content"), (str, type(None))) for x in m))

def full_key(m):
    u = next((x["content"] for x in m if x["role"]=="user"), "")
    a = next((x["content"] for x in reversed(m) if x["role"]=="assistant"), "")
    return hashlib.md5((norm(u)+"§"+norm(a)).encode()).hexdigest()

master, report = [], {}
fixes, bad_total = Counter(), 0
hash_src = {}                     # full-hash → اولین source

for fp, expected_src in FILES:
    rows, bad, meta_fill = [], 0, 0
    for l in open(fp, encoding="utf-8"):
        r = json.loads(l)
        m = r.get("messages")
        if not valid_msgs(m): bad += 1; continue
        if not valid_msgs(m): bad += 1; continue
        # ---- fixهای ترمیمی (محتوا/متادیتا، نه حذف) ----
        if r.get("source") == "Persian-Thinking":
            dev = next((x["content"] for x in m if x["role"]=="system"), "")
            u   = next((x["content"] for x in m if x["role"]=="user"), "")
            if u.lstrip().lower().startswith("system:"):
                body = u.lstrip()[7:].lstrip()
                if dev and body.startswith(dev[:80]):
                    body = body[len(dev):].lstrip(" :\n-–—")
                    for x in m:
                        if x["role"] == "user": x["content"] = body
                    fixes["psthink_strip"] += 1
                else:
                    fixes["psthink_manual_check"] += 1
        if r.get("source") == "Persian-Alpaca-Reasoning":
            r["original_language"] = "en"; fixes["alpaca_origlang"] += 1
        if fp == "jumpshift_extra_200.jsonl" and not r.get("source_subset"):
            r["source_subset"] = "topup"
        if not r.get("source"):
            r["source"] = expected_src; meta_fill += 1
        r.setdefault("qc_flags", [])
        rows.append(r)
    bad_total += bad
    report[fp] = {"rows": len(rows), "bad_msgs": bad, "meta_filled": meta_fill}
    master.extend(rows)
    print(f"{fp:<42} rows={len(rows):>6,}  bad={bad}  meta_filled={meta_fill}")

print(f"\nfixes: {dict(fixes)} | bad_msgs total: {bad_total}")

# ---- پرچم‌گذاری exact dup کراس-منبع (بدون حذف) ----
dup_matrix = Counter()
for r in master:
    k = full_key(r["messages"])
    if k in hash_src:
        r["qc_flags"].append("exact_dup_cross_source")
        dup_matrix[f"{hash_src[k]} ↔ {r['source']}"] += 1
    else:
        hash_src[k] = r["source"]

print(f"\nflagged exact dups: {sum(dup_matrix.values()):,}")
print("dup matrix:")
for k, v in dup_matrix.most_common(15): print(f"  {k}: {v}")

with open("master_stage1.jsonl", "w", encoding="utf-8") as f:
    for r in master:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

src_c = Counter(r["source"] for r in master)
th_c  = sum(1 for r in master if r.get("thinking"))
lens  = sorted(r.get("length_chars") or sum(len(x.get("content") or "") for x in r["messages"])
               for r in master)
p = lambda q: lens[int(q*(len(lens)-1))]
print(f"\n===== MASTER STAGE 1 =====")
print(f"total: {len(master):,} | thinking: {th_c:,} ({100*th_c/len(master):.1f}%)")
print(f"len_chars p50/p90/p99/max: {p(.5):,} / {p(.9):,} / {p(.99):,} / {lens[-1]:,}")
print("per-source:")
for s, c in src_c.most_common(): print(f"  {s:<28} {c:>7,}")
json.dump({"stage1_rows": len(master), "fixes": dict(fixes),
           "dup_matrix": dict(dup_matrix), "files": report},
          open("merge_stage1_manifest.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved: master_stage1.jsonl + merge_stage1_manifest.json")