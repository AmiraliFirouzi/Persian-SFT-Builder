# ============================================================
# Selection — Persian-Thinking (v3 —  developer/user/analysis/final)
# ============================================================
import hashlib, json, random, re
from collections import Counter
from datasets import load_dataset

REPO = "artindnr/Persian-Thinking"  
raw = load_dataset(REPO)
ds = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
print("total rows:", len(ds))
assert len(ds) != 2147, "❌ این Alpaca است!"

TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

def extract(ex):
    """اولویت: فیلدهای top-level (developer/user/analysis/final)؛ fallback: messages"""
    sys_p = str(ex.get("developer") or "")
    user  = str(ex.get("user") or "")
    final = str(ex.get("final") or "")
    rea   = str(ex.get("analysis") or "")
    m = ex.get("messages") if isinstance(ex.get("messages"), list) else []
    if not sys_p and m and m[0].get("role") == "system":
        sys_p = str(m[0].get("content") or "")
    if not user:
        user = next((str(x.get("content") or "") for x in m if x.get("role") == "user"), "")
    if not final:
        final = next((str(x.get("content") or "") for x in reversed(m)
                      if x.get("role") == "assistant"), "")
    if not rea:
        rea = next((str(x.get("thinking")) for x in reversed(m)
                    if x.get("role") == "assistant" and x.get("thinking")), "")
        if not rea and "<think>" in final:
            parts = re.split(r"</?think>", final)
            rea, final = parts[1].strip(), parts[-1].strip()
    return sys_p, user, final, rea, len(m)

rows, seen, stats = [], set(), Counter()
turns_dist = Counter()
for ex in ds:
    stats["total"] += 1
    sys_p, user, final, rea, turns = extract(ex)
    turns_dist[turns] += 1
    if len(user) < 10 or len(final) < 10 or len(rea) < 50:
        stats["skipped"] += 1; continue
    h = hashlib.md5((norm(user) + "§" + norm(final)).encode()).hexdigest()
    if h in seen: stats["dups"] += 1; continue
    seen.add(h)
    stats["ok"] += 1
    rows.append({"sys": sys_p, "user": user, "final": final, "rea": rea,
                 "length_chars": len(sys_p) + len(user) + len(final) + len(rea)})

print("\n===== REPORT =====")
print(dict(stats), "| kept =", len(rows))
print("messages-per-sample:", dict(sorted(turns_dist.items())))
has_sys = sum(1 for r in rows if r["sys"])
print(f"with system prompt: {has_sys}/{len(rows)}")

random.Random(42).shuffle(rows)
with open("persian_thinking_selected.jsonl", "w", encoding="utf-8") as f:
    for n, r in enumerate(rows):
        msgs = ([{"role": "system", "content": r["sys"]}] if r["sys"] else []) + \
               [{"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["final"]}]
        f.write(json.dumps({
            "id": f"psthink-{n:05d}", "source": "Persian-Thinking",
            "source_subset": "thinking", "task": "reasoning", "language": "fa",
            "thinking": True, "synthetic_or_human": "synthetic_translated",
            "original_language": "en", "length_chars": r["length_chars"],
            "quality_score": None, "reasoning": r["rea"], "messages": msgs,
        }, ensure_ascii=False) + "\n")
print(f"\nSAVED: persian_thinking_selected.jsonl ({len(rows):,} rows)")


r = rows[0]
print(f"\nSYSTEM: {r['sys'][:150]}\n\nU: {r['user'][:200]}\n\nTHINK (کامل):\n{r['rea'][:700]}\n\nA: {r['final'][:200]}")