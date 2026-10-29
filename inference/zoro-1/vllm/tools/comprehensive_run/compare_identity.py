"""Byte-identity check: compare per-item generations of a candidate run job vs a baseline run job (same client, greedy)."""
import json, sys, glob, os
def load(d):
    f = glob.glob(os.path.join(d, "gsm8k_details_*.json"))[0]
    return {x["index"]: x for x in json.load(open(f))}
base, cand = load(sys.argv[1]), load(sys.argv[2])
shared = sorted(set(base) & set(cand)); same = 0; diffs = []
for i in shared:
    if base[i]["generation"] == cand[i]["generation"]:
        same += 1
    else:
        a, b = base[i]["generation"], cand[i]["generation"]; k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        diffs.append((i, k, len(a), len(b), base[i]["correct"], cand[i]["correct"]))
print(f"shared items {len(shared)}: identical generations {same}, differing {len(diffs)}")
for d in diffs[:10]:
    print(f"  item {d[0]}: first diff at char {d[1]} (len base {d[2]} / cand {d[3]}); correct base={d[4]} cand={d[5]}")
acc_b = sum(base[i]["correct"] for i in shared) / len(shared); acc_c = sum(cand[i]["correct"] for i in shared) / len(shared)
print(f"accuracy on shared: base {100*acc_b:.1f}%  cand {100*acc_c:.1f}%")
