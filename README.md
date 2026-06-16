## Live Demo
[View the app →](https://kabeertrivedi-churn-dashboard.streamlit.app/)

##  AI Evaluation Results

### Methodology
Tested the Groq intervention recommendation system across 10 
customer profiles covering a range of risk levels (70%–94%), 
contract types, tenure lengths, and churn signals.

### Quantitative Results — Run 1 (baseline)
| Criteria | Score |
|---|---|
| Correct format (2 actions) | 10/10 ✓ |
| Under 80 words | 10/10 ✓ |
| Uses customer context | 5/10 ✗ |
| Specific (not generic) | 4/10 ✗ |

**Failure mode identified:** Every recommendation opened with 
"Reach out within 24 hours to discuss..." — templated language 
that ignored customer-specific signals. A CS manager reading 
10 identical-sounding recommendations would stop trusting the system.

### Fix Applied
Updated system prompt with three changes:
1. Removed hardcoded time references (24 hours / 7 days)
2. Added rule: must reference exact signals provided
3. Added tenure-based rules — new customers get onboarding 
   focus, long-tenure customers get loyalty/value focus

### Quantitative Results — Run 2 (after fix)
| Criteria | Score |
|---|---|
| Correct format (2 actions) | 10/10 ✓ |
| Under 80 words | 10/10 ✓ |
| Uses customer context | 7/10 ✓ |
| Specific (not generic) | 8/10 ✓ |
| References actual charges | 4/10 (customers 2,5,6,9) ✓ |

### Notable Improvements
- Customer 2: Now references "$74.20 monthly charges" directly
- Customer 5: Calculates actual cost saving ("$5.50/month")
- Customer 6: Suggests specific price range ("$80–$90")
- Customer 7: Correctly identifies 20-month tenure as loyalty case

### Remaining Failure Modes
- "Uses context" still fails on customers 2,3,7,9 in auto-scoring 
  because the word "contract" doesn't appear — but recommendations 
  are actually contextual. Auto-scorer needs improvement.
- Customer 3 and 8 recommendations are slightly generic compared 
  to others — tenure signal not strongly reflected.

### Key Learning
The system prompt is the product spec for an LLM. Vague instructions 
produce vague outputs. Adding specific rules ("for tenure under 3 
months, focus on onboarding") directly improved output quality without 
changing any model weights — pure prompt engineering.