# ============================================================
# Collection — Replay از smoltalk2 (config SFT) → replay_selected.jsonl
# streaming + reservoir + dedup کراس-subset
# ============================================================
import hashlib, json, random, re
from collections import Counter
from datasets import load_dataset

REPO, SEED = "HuggingFaceTB/smoltalk2", 42
rng = random.Random(SEED)
TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

THINK = {  # جمع = 8,500
 "OpenThoughts3_1.2M_think": (4200, "math_code_reasoning"),
 "smoltalk_everyday_convs_reasoning_Qwen3_32B_think": (1300, "reasoning_conversation"),
 "multi_turn_reasoning_if_think": (1000, "reasoning_if"),
 "smoltalk_systemchats_Qwen3_32B_think": (600, "systemchat"),
 "smoltalk_multilingual8_Qwen3_32B_think": (400, "multilingual"),
 "aya_dataset_Qwen3_32B_think": (400, "multilingual"),
 "smolagents_toolcalling_traces_think": (250, "tool_calling"),
 "s1k_1.1_think": (200, "reasoning"),
 "table_gpt_Qwen3_32B_think": (150, "table"),
}
NOTHINK = {  # جمع = 8,802
 "smoltalk_smollm3_smol_magpie_ultra_no_think": (2000, "general_if"),
 "tulu_3_sft_personas_instruction_following_no_think": (1500, "instruction_following"),
 "OpenHermes_2.5_no_think": (1400, "general"),
 "smoltalk_smollm3_everyday_conversations_no_think": (700, "conversation"),
 "smoltalk_smollm3_systemchats_30k_no_think": (600, "systemchat"),
 "smoltalk_smollm3_smol_summarize_no_think": (450, "summarize"),
 "smoltalk_smollm3_smol_rewrite_no_think": (400, "rewrite"),
 "smoltalk_multilingual_8languages_lang_5_no_think": (400, "multilingual"),
 "smoltalk_smollm3_explore_instruct_rewriting_no_think": (300, "rewrite"),
 "hermes_function_calling_v1_no_think": (350, "tool_calling"),
 "Mixture_of_Thoughts_science_no_think": (300, "science"),
 "xlam_traces_no_think": (250, "tool_calling"),
 "table_gpt_no_think": (152, "table"),
}
assert sum(q for q, _ in THINK.values()) == 8500
assert sum(q for q, _ in NOTHINK.values()) == 8802

MAX_CHARS = 24_000                    # سقف طول نمونه (~6-7K توکن)
ALLOWED = {"system", "user", "assistant", "tool"}   # tool برای function-calling
SCAN_CAP_BIG = 150_000                # برای subsetهای بزرگ (مثل OpenThoughts3)

def valid(msgs):
    if not isinstance(msgs, list) or len(msgs) < 2: return False, "structure"
    if not all(m.get("role") in ALLOWED for m in msgs): return False, "role"
    if not any(m["role"] == "user" for m in msgs): return False, "no_user"
    if not any(m["role"] == "assistant" for m in msgs): return False, "no_assistant"
    L = sum(len(str(m.get("content") or "")) for m in msgs)
    if L < 50: return False, "too_short"
    if L > MAX_CHARS: return False, "too_long"
    return True, L

seen, all_rows = set(), []
print(f"{'subset':<55} {'scanned':>8} {'kept':>6}  drops")
for thinking, quotas in [(True, THINK), (False, NOTHINK)]:
    for sub, (k, task) in quotas.items():
        cap = SCAN_CAP_BIG if k >= 1000 else None
        stream = load_dataset(REPO, "SFT", split=sub, streaming=True)
        res, scanned, valid_n = [], 0, 0
        drops = Counter()
        for ex in stream:
            if cap is not None and scanned >= cap: break
            scanned += 1
            ok, why = valid(ex.get("messages"))
            if not ok: drops[why] += 1; continue
            valid_n += 1
            h = hashlib.md5("§".join(
                norm(str(m.get("content") or "")) for m in ex["messages"]
            ).encode()).hexdigest()
            if h in seen: drops["dup"] += 1; continue
            if len(res) < k:
                res.append((h, ex["messages"])); seen.add(h)
            else:
                j = rng.randint(0, valid_n - 1)
                if j < k:
                    old = res[j]; res[j] = (h, ex["messages"])
                    seen.discard(old[0]); seen.add(h)
        all_rows.extend((sub, task, thinking, m) for _, m in res)
        print(f"{sub:<55} {scanned:>8,} {len(res):>6}  {dict(drops) or ''}")

short = [(s, THINK.get(s, NOTHINK.get(s))[0])
         for s, *_, in [] ] # placeholder
per = Counter(s for s, *_ in all_rows)
need = {**{s: q for s, (q, _) in THINK.items()}, **{s: q for s, (q, _) in NOTHINK.items()}}
shortfall = {s: (need[s], per[s]) for s in need if per[s] < need[s]}
print(f"\nTOTAL: {len(all_rows):,} / 17,302   SHORTFALL: {shortfall or 'none'}")

rng.shuffle(all_rows)
with open("replay_selected.jsonl", "w", encoding="utf-8") as f:
    for n, (sub, task, th, msgs) in enumerate(all_rows):
        L = sum(len(str(m.get("content") or "")) for m in msgs)
        f.write(json.dumps({
            "id": f"replay-{n:05d}", "source": "Replay-smoltalk2",
            "source_subset": sub, "task": task, "language": "en",
            "thinking": th, "synthetic_or_human": "synthetic",
            "original_language": "en", "length_chars": L,
            "quality_score": None, "messages": msgs,
        }, ensure_ascii=False) + "\n")
print(f"SAVED: replay_selected.jsonl ({len(all_rows):,} rows)")

th_c = Counter(t for _, _, t, _ in all_rows)
print("think/no_think:", dict(th_c))
for sub, task, th, msgs in all_rows[:2]:
    print(f"\n--- {sub} (thinking={th}) ---")
    for m in msgs[:3]: print(f"{m['role']}: {str(m.get('content'))[:130]}")




part 2 :



# ============================================================
# Replay top-up — OpenThoughts3 (پنجره 150K به بعد)
# ============================================================
import hashlib, json, random, re
from collections import Counter
from datasets import load_dataset

REPO, SEED = "HuggingFaceTB/smoltalk2", 42
rng = random.Random(SEED)
TR = str.maketrans("يكةۀأإؤئ", "یکههآایئ")
def norm(s): return re.sub(r"[\u200c\s]+", " ", str(s).lower().translate(TR)).strip()

NEED      = 17302 - 15956      # 1,346
MAX_CHARS = 24_000
ALLOWED   = {"system", "user", "assistant", "tool"}
SUB, SKIP, SCAN = "OpenThoughts3_1.2M_think", 150_000, 120_000

def h_of(msgs):
    return hashlib.md5("§".join(
        norm(str(m.get("content") or "")) for m in msgs).encode()).hexdigest()

def valid(msgs):
    if not isinstance(msgs, list) or len(msgs) < 2: return False
    if not all(m.get("role") in ALLOWED for m in msgs): return False
    if not any(m["role"] == "user" for m in msgs): return False
    if not any(m["role"] == "assistant" for m in msgs): return False
    L = sum(len(str(m.get("content") or "")) for m in msgs)
    return 50 <= L <= MAX_CHARS

# ۱) فایل موجود + بازسازی seen
rows = [json.loads(l) for l in open("replay_selected.jsonl", encoding="utf-8")]
seen = {h_of(r["messages"]) for r in rows}
print(f"existing: {len(rows):,} | need: {NEED:,}")

# ۲) اسکن پنجره بعدی با reservoir
stream = load_dataset(REPO, "SFT", split=SUB, streaming=True).skip(SKIP)
res, scanned, valid_n = [], 0, 0
for ex in stream:
    if scanned >= SCAN: break
    scanned += 1
    msgs = ex.get("messages")
    if not valid(msgs): continue
    valid_n += 1
    h = h_of(msgs)
    if h in seen: continue
    if len(res) < NEED:
        res.append((h, msgs)); seen.add(h)
    else:
        j = rng.randint(0, valid_n - 1)
        if j < NEED:
            old = res[j]; res[j] = (h, msgs)
            seen.discard(old[0]); seen.add(h)

print(f"scanned: {scanned:,} | valid: {valid_n:,} | got: {len(res)}")

# ۳) merge + reshuffle + ذخیره با idهای نو
rows.extend({"messages": m} for _, m in res)
rng.shuffle(rows)
with open("replay_selected.jsonl", "w", encoding="utf-8") as f:
    for n, r in enumerate(rows):
        r["id"] = f"replay-{n:05d}"
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

th = sum(1 for r in rows if r["thinking"])
per = Counter(r["source_subset"] for r in rows)
print(f"\nFINAL: {len(rows):,} / 17,302 | think={th}  no_think={len(rows)-th}")
print(f"OpenThoughts3 total now: {per['OpenThoughts3_1.2M_think']}")



part 3 : 


# ============================================================
# Repair — پر کردن متادیتای رکوردهای top-up (بدون re-download)
# ============================================================
import json, random
from collections import Counter

rng = random.Random(42)

rows = [json.loads(l) for l in open("replay_selected.jsonl", encoding="utf-8")]
orphans = [r for r in rows if "thinking" not in r]
print(f"total: {len(rows):,} | orphans: {len(orphans):,}")
assert len(orphans) == 1346, "تعداد یتیم‌ها با انتظار نمی‌خواند — خروجی را بفرست"

for r in orphans:
    r.update({
        "source": "Replay-smoltalk2",
        "source_subset": "OpenThoughts3_1.2M_think",
        "task": "math_code_reasoning",
        "language": "en", "thinking": True,
        "synthetic_or_human": "synthetic", "original_language": "en",
        "length_chars": sum(len(str(m.get("content") or "")) for m in r["messages"]),
        "quality_score": None,
    })

rng.shuffle(rows)
with open("replay_selected.jsonl", "w", encoding="utf-8") as f:
    for n, r in enumerate(rows):
        r["id"] = f"replay-{n:05d}"
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

th = sum(1 for r in rows if r["thinking"])
per = Counter(r["source_subset"] for r in rows)
print(f"\nFINAL: {len(rows):,} / 17,302 | think={th}  no_think={len(rows)-th}")
print(f"OpenThoughts3 total now: {per['OpenThoughts3_1.2M_think']:,}")
print("keys check:", sorted(set(k for r in rows for k in r) - {"messages"}))