import json

d = json.load(open("test_output.json"))

for s in d["scenarios"]:
    tag = "PASS" if s["pass"] else "FAIL"
    print(f'{tag} {s["prob"]:.3f} {s["name"]}')

print()
t = d["tenure_monotonicity"]
print(f'tenure_mono={t["decreasing"]}')
for v in t["values"]:
    print(f'  t={v["tenure"]} p={v["prob"]}')

print()
c = d["charges_monotonicity"]
print(f'charges_mono={c["increasing"]}')
for v in c["values"]:
    print(f'  c={v["charges"]} p={v["prob"]}')

print()
for v in d["contract_impact"]:
    print(f'  ct={v["contract"]} p={v["prob"]}')
