# Product Requirements Document
## RetainWiseIQ — AI-Powered Customer Retention Intelligence Platform

**Version:** 1.0  
**Author:** Kabeer Trivedi  
**Date:** June 2026  
**Status:** Shipped

---

## 1. Problem Statement

B2B SaaS companies lose 5–7% of customers monthly without warning.
Customer Success teams discover churn after it happens — not before.
This results in lost revenue, reactive winback campaigns, and wasted
CS team time on the wrong accounts.

**The gap:** No early warning system exists that tells a CS manager
which specific customers are at risk, why they're at risk, and
exactly what to do about it — in plain English.

---

## 2. Goal

Transform reactive churn management into proactive retention by
predicting which customers will churn 30 days in advance and
generating personalised AI intervention recommendations for each
at-risk account.

---

## 3. Target User

**Primary:** Customer Success Manager (CSM) at a 50–500 person
B2B SaaS company.

**Profile:**
- Non-technical — needs insights, not model outputs
- Manages 50–150 accounts simultaneously
- Currently relies on gut feel or lagging indicators (support
  tickets, NPS) to identify at-risk accounts
- Needs to know who to call today and what to say

---

## 4. Success Metrics

Defined before building — not after.

| Metric | Target | Result |
|---|---|---|
| Model AUC-ROC | ≥ 0.80 | 0.842 ✓ |
| Churn recall | ≥ 0.70 | 0.78 ✓ |
| AI format compliance | 10/10 | 10/10 ✓ |
| AI specificity (run 2) | ≥ 7/10 | 8/10 ✓ |
| Dashboard load time | < 10 seconds | ~6 seconds ✓ |

---

## 5. Scope

### In Scope — V1
- Churn probability prediction per customer
- Risk classification (Low / Medium / High / Critical)
- SHAP-based explainability showing top churn drivers
- AI-generated intervention recommendations per customer
- Interactive risk threshold filtering
- Contract type churn rate breakdown
- Deployed public URL

### Out of Scope — V1
- CRM integration (Salesforce, HubSpot)
- Real-time data ingestion
- Multi-tenant authentication
- Email automation triggered by risk alerts
- A/B testing of intervention playbooks
- Mobile app

---

## 6. AI Model Spec

| Component | Decision | Rationale |
|---|---|---|
| Algorithm | XGBoost Classifier | Interpretable, fast, industry standard for tabular data |
| Explainability | SHAP TreeExplainer | Shows per-customer feature contributions in plain English |
| Class imbalance | scale_pos_weight=2.7 | 73/27 class split — prevents model predicting all No-churn |
| Train/test split | 80/20 stratified | Honest evaluation on unseen data |
| LLM provider | Groq (Llama 3.3 70B) | Free tier, fast inference, sufficient quality for recommendations |

---

## 7. Key Product Decisions

**Why XGBoost over a neural network?**
Interpretability. A CS manager can't act on "the model said so."
SHAP makes every prediction explainable — "this customer is at risk
because of month-to-month contract + low tenure + high charges."

**Why SHAP over feature importance?**
Global feature importance tells you what matters across all customers.
SHAP tells you what matters for this specific customer — which is
what a CS manager needs to prioritise their outreach.

**Why cache the LLM recommendations?**
Without caching, every dashboard interaction triggers an API call.
With @st.cache_data(ttl=3600), the same customer profile returns
the same recommendation for 1 hour — reducing API costs by ~90%
during normal dashboard use.

**Why Groq over OpenAI/Anthropic?**
Free tier with no credit card required. For a portfolio project,
minimising cost barriers makes the project reproducible by anyone
who clones the repo.

---

## 8. AI Evaluation Framework

### Methodology
Tested LLM recommendations across 10 customer profiles covering:
risk levels 70–94%, all contract types, tenure 1–20 months.

### Run 1 — Baseline
| Criteria | Score |
|---|---|
| Format compliance | 10/10 |
| Under 80 words | 10/10 |
| Uses customer context | 5/10 |
| Specific recommendations | 4/10 |

**Failure mode:** Every recommendation opened with identical
templated language regardless of customer profile.

### Fix Applied
Updated system prompt with three specific rules:
1. Must reference exact signals provided for each customer
2. Removed hardcoded time references
3. Added tenure-based rules (new vs long-term customers)

### Run 2 — After Fix
| Criteria | Score |
|---|---|
| Format compliance | 10/10 |
| Under 80 words | 10/10 |
| Uses customer context | 7/10 |
| Specific recommendations | 8/10 |

**Key learning:** The system prompt is the product spec for an LLM.
Vague instructions produce vague outputs. Specificity in prompts
directly improves output quality without changing any model weights.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Model flags healthy customers | Show confidence score, only surface >70% probability |
| Generic AI recommendations | Eval framework with specificity scoring, iterated prompt |
| API costs in production | Caching layer (ttl=3600), batch processing for bulk analysis |
| Data privacy (DPDP Act 2023) | Demo uses public IBM dataset only, PII anonymisation required for production |
| Model drift over time | Retrain monthly, monitor AUC on new data |

---

## 10. V2 Roadmap

If productionising this, next priorities would be:

1. **CRM integration** — pull live customer data from Salesforce/HubSpot
2. **Automated alerts** — Slack/email when a customer crosses risk threshold
3. **Intervention tracking** — did the CS action actually prevent churn?
4. **A/B test playbooks** — measure which intervention type works best
5. **Multi-tenant auth** — separate dashboards per CS team member

---

## 11. Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost + scikit-learn |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Charts | Plotly |
| LLM | Groq API (Llama 3.3 70B) |
| Data | IBM Telco Churn Dataset (Kaggle) |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---

*This PRD was written as part of an  project
demonstrating end-to-end product thinking on an AI system —
from problem definition through model evaluation to deployment.*