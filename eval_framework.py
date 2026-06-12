import json
from ai_layer import get_intervention
print("Starting eval framework...")

#  10 test cases covering different risk profiles 
TEST_CUSTOMERS = [
    {"churn_risk": 94, "contract": "Month-to-month", "tenure_months": 2,
     "monthly_charges": 89.50, "top_drivers": "Contract type, Low tenure"},
    {"churn_risk": 87, "contract": "Month-to-month", "tenure_months": 8,
     "monthly_charges": 74.20, "top_drivers": "No online security, High monthly charges"},
    {"churn_risk": 81, "contract": "Month-to-month", "tenure_months": 14,
     "monthly_charges": 95.30, "top_drivers": "No tech support, High monthly charges"},
    {"churn_risk": 76, "contract": "One year", "tenure_months": 5,
     "monthly_charges": 65.00, "top_drivers": "Low tenure, Contract type"},
    {"churn_risk": 72, "contract": "Month-to-month", "tenure_months": 3,
     "monthly_charges": 45.50, "top_drivers": "Contract type, Low tenure"},
    {"churn_risk": 91, "contract": "Month-to-month", "tenure_months": 1,
     "monthly_charges": 110.00, "top_drivers": "Contract type, High monthly charges"},
    {"churn_risk": 83, "contract": "Month-to-month", "tenure_months": 20,
     "monthly_charges": 88.75, "top_drivers": "No online security, No tech support"},
    {"churn_risk": 78, "contract": "One year", "tenure_months": 11,
     "monthly_charges": 58.20, "top_drivers": "Contract type, Low tenure"},
    {"churn_risk": 70, "contract": "Month-to-month", "tenure_months": 6,
     "monthly_charges": 72.00, "top_drivers": "High monthly charges, No tech support"},
    {"churn_risk": 85, "contract": "Month-to-month", "tenure_months": 4,
     "monthly_charges": 99.99, "top_drivers": "Contract type, High monthly charges, Low tenure"},
]

#  Evaluation criteria 
def evaluate_recommendation(rec: str, customer: dict) -> dict:
    """Scoring a recommendation on 4 criteria. """
    
    word_count = len(rec.split())
    has_two_actions = "1." in rec and "2." in rec
    mentions_contract = customer["contract"].lower() in rec.lower() or \
                        "contract" in rec.lower()

    return {
        "word_count": word_count,
        "under_80_words": word_count <= 80,
        "has_two_actions": has_two_actions,
        "mentions_context": mentions_contract,
        "is_specific": None,    
        "is_actionable": None,  
        "quality_score": None,  
    }


#  Running the eval 
print("Running evaluation across 10 customers...\n")
print("=" * 60)

results = []
for i, customer in enumerate(TEST_CUSTOMERS):
    print(f"\nCustomer {i+1}/{len(TEST_CUSTOMERS)}")
    print(f"Profile: {customer['churn_risk']}% risk | {customer['contract']} | {customer['tenure_months']}mo tenure")
    print(f"Signals: {customer['top_drivers']}")
    print("-" * 40)
    
    rec = get_intervention(customer)
    scores = evaluate_recommendation(rec, customer)
    
    print(f"RECOMMENDATION:\n{rec}")
    print(f"\nAUTO SCORES:")
    print(f"  Words: {scores['word_count']} ({'✓' if scores['under_80_words'] else '✗ over limit'})")
    print(f"  2 actions: {'✓' if scores['has_two_actions'] else '✗'}")
    print(f"  Uses context: {'✓' if scores['mentions_context'] else '✗'}")
    print("\n  → YOUR RATING: Is this specific? Actionable? Score 1-5?")
    print("  → Note any failure modes below:")
    print("=" * 60)
    
    results.append({"customer": customer, "recommendation": rec, "scores": scores})

# Saving results to file for your README documentation
with open("eval_results.txt", "w") as f:
    f.write("RETAINIQ — AI EVAL RESULTS\n")
    f.write("="*60 + "\n")
    f.write("Instructions: Open this file, read each recommendation,\n")
    f.write("and fill in: is_specific, is_actionable, quality_score (1-5)\n")
    f.write("="*60 + "\n\n")
    for i, r in enumerate(results):
        f.write(f"\n--- Customer {i+1} ---\n")
        f.write(f"Profile: {r['customer']}\n")
        f.write(f"Recommendation:\n{r['recommendation']}\n")
        f.write(f"Auto scores: {r['scores']}\n")
        f.write(f"\nYOUR RATING:\n")
        f.write(f"  is_specific   : [ ] True  [ ] False\n")
        f.write(f"  is_actionable : [ ] True  [ ] False\n")
        f.write(f"  quality_score : ___ /5\n")
        f.write(f"  notes         : ___________________________\n")
        f.write("\n" + "-"*60 + "\n")

print("\nResults saved to eval_results.txt")
print("Open the file, read each recommendation, fill in your ratings.")


