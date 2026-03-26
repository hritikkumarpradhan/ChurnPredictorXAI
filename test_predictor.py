"""Compact test — write results as JSON for easy reading."""
import requests, json

URL = "http://127.0.0.1:5000/api/predict"

def p(d):
    return requests.post(URL, json=d).json()

results = {}

# === 1. Scenario Tests ===
scenarios = {
    "high_risk_short_tenure_mtm_fiber": {
        "data": {"tenure":2,"monthly_charges":110,"total_charges":220,"contract_type":"Month-to-month","internet_service":"Fiber optic","tech_support":"No"},
        "expect": "HIGH"
    },
    "low_risk_long_tenure_2yr_dsl_support": {
        "data": {"tenure":60,"monthly_charges":30,"total_charges":1800,"contract_type":"Two year","internet_service":"DSL","tech_support":"Yes"},
        "expect": "LOW"
    },
    "edge_zero_tenure_max_charges": {
        "data": {"tenure":0,"monthly_charges":120,"total_charges":0,"contract_type":"Month-to-month","internet_service":"Fiber optic","tech_support":"No"},
        "expect": "VERY_HIGH"
    },
    "edge_max_tenure_min_charges_2yr": {
        "data": {"tenure":72,"monthly_charges":20,"total_charges":1440,"contract_type":"Two year","internet_service":"No","tech_support":"Yes"},
        "expect": "VERY_LOW"
    },
    "medium_mid_tenure_1yr_fiber": {
        "data": {"tenure":24,"monthly_charges":70,"total_charges":1680,"contract_type":"One year","internet_service":"Fiber optic","tech_support":"No"},
        "expect": "MEDIUM"
    },
    "borderline_mtm_dsl_support": {
        "data": {"tenure":36,"monthly_charges":65,"total_charges":2340,"contract_type":"Month-to-month","internet_service":"DSL","tech_support":"Yes"},
        "expect": "MEDIUM"
    },
    "no_internet_long_tenure": {
        "data": {"tenure":48,"monthly_charges":25,"total_charges":1200,"contract_type":"One year","internet_service":"No","tech_support":"No internet"},
        "expect": "LOW"
    },
    "new_fiber_1yr": {
        "data": {"tenure":3,"monthly_charges":95,"total_charges":285,"contract_type":"One year","internet_service":"Fiber optic","tech_support":"No"},
        "expect": "HIGH"
    },
}

scenario_results = []
for name, cfg in scenarios.items():
    res = p(cfg["data"])
    prob = res["churn_probability"]
    exp = cfg["expect"]
    ok = True
    if "VERY_HIGH" in exp and prob < 0.7: ok = False
    elif "VERY_LOW" in exp and prob > 0.3: ok = False
    elif exp == "HIGH" and prob < 0.5: ok = False
    elif exp == "LOW" and prob > 0.5: ok = False
    scenario_results.append({"name": name, "prob": prob, "pred": res["prediction"], "expect": exp, "pass": ok})

results["scenarios"] = scenario_results

# === 2. Tenure monotonicity ===
tenure_probs = []
for t in [0, 12, 24, 36, 48, 60, 72]:
    res = p({"tenure":t,"monthly_charges":80,"total_charges":t*80,"contract_type":"Month-to-month","internet_service":"Fiber optic","tech_support":"No"})
    tenure_probs.append({"tenure": t, "prob": res["churn_probability"]})

results["tenure_monotonicity"] = {
    "values": tenure_probs,
    "decreasing": all(tenure_probs[i]["prob"] >= tenure_probs[i+1]["prob"] for i in range(len(tenure_probs)-1))
}

# === 3. Charges monotonicity ===
charge_probs = []
for mc in [20, 40, 60, 80, 100, 120]:
    res = p({"tenure":12,"monthly_charges":mc,"total_charges":12*mc,"contract_type":"Month-to-month","internet_service":"Fiber optic","tech_support":"No"})
    charge_probs.append({"charges": mc, "prob": res["churn_probability"]})

results["charges_monotonicity"] = {
    "values": charge_probs,
    "increasing": all(charge_probs[i]["prob"] <= charge_probs[i+1]["prob"] for i in range(len(charge_probs)-1))
}

# === 4. Contract impact ===
contract_probs = []
for ct in ["Month-to-month", "One year", "Two year"]:
    res = p({"tenure":12,"monthly_charges":80,"total_charges":960,"contract_type":ct,"internet_service":"Fiber optic","tech_support":"No"})
    contract_probs.append({"contract": ct, "prob": res["churn_probability"]})

results["contract_impact"] = contract_probs

# Write
with open("test_output.json", "w") as f:
    json.dump(results, f, indent=2)

print("DONE - wrote test_output.json")
