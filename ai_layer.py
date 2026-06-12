from groq import Groq
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or "MISSING_KEY"
client = Groq(
    api_key=api_key
)

#System prompt 
SYSTEM_PROMPT = """You are a Customer Success advisor at a B2B SaaS company.
Your job is to recommend specific, actionable interventions for 
at-risk customers based on their usage data and churn signals.

Rules:
- Be specific and practical — no generic advice
- Keep response under 80 words
- Give exactly 2 recommendations
- Format as:
- Format as:
  1. [Immediate action — be specific about what to say and do]
  2. [Follow-up action — be specific about the next step]
- Do not mention the AI model or predictions
- Write as if briefing a Customer Success Manager
- Each recommendation must reference the specific signals provided
- Never give generic advice that could apply to any customer
- For long tenure customers (>12 months), focus on loyalty and value. For new customers (<3 months), focus on onboarding and quick wins"""

#  Main recommendation function 
def get_intervention(customer_data: dict) -> str:
    """
    Generate an intervention recommendation for one at-risk customer.
    
    customer_data keys:
        churn_risk, contract, tenure_months,
        monthly_charges, top_drivers
    """
    
    user_prompt = f"""Customer profile:
- Churn risk: {customer_data['churn_risk']}%
- Contract type: {customer_data['contract']}
- Tenure: {customer_data['tenure_months']} months
- Monthly charges: ${customer_data['monthly_charges']:.2f}
- Key churn signals: {customer_data['top_drivers']}

What are the 2 most important interventions for this customer?"""

    if api_key == "MISSING_KEY":
        return "⚠️ Error: API key is missing. Please add GROQ_API_KEY to your .env file and restart the app."

    try:
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        return message.choices[0].message.content

    except Exception as e:
        return f"Error generating recommendation: {str(e)}"


#  Batch function for eval framework 
def get_batch_interventions(customers: list) -> list:
    """
    Generate recommendations for a list of customers.
    Returns list of dicts with customer data + recommendation.
    Used for the eval framework.
    """
    results = []
    for customer in customers:
        rec = get_intervention(customer)
        results.append({
            **customer,
            "recommendation": rec
        })
    return results


#  Streamlit cached version 
@st.cache_data(ttl=3600)
def get_cached_intervention(
    churn_risk: float,
    contract: str,
    tenure_months: int,
    monthly_charges: float,
    top_drivers: str
) -> str:
    """
    Cached version — same customer data returns same result
    for 1 hour. Prevents unnecessary API calls while browsing
    the dashboard.
    """
    customer_data = {
        "churn_risk": churn_risk,
        "contract": contract,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "top_drivers": top_drivers
    }
    return get_intervention(customer_data)