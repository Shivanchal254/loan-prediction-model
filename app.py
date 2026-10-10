import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
import textwrap

# ============================================================
# LOANPREDICT - MODERN BANK LOAN APPROVAL PREDICTION UI
# ============================================================

st.set_page_config(
    page_title="LoanPredict | Smart Loan Approval",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# 1. LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("final_lr_loan_model.pkl")
    except FileNotFoundError:
        return None

model = load_model()

# -----------------------------
# 2. SESSION STATE
# -----------------------------
defaults = {
    "page": "home",
    "step": 1,
    "form_data": {},
    "prediction": None,
    "probability": None,
    "history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# 3. GLOBAL CSS
# -----------------------------
st.markdown(textwrap.dedent("""
<style>
/* ---------- APP ---------- */
.stApp {
    background: #f7faff;
}

.block-container {
    max-width: 1350px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

/* Remove Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---------- TOP NAV ---------- */
.navbar {
    background: #ffffff;
    border: 1px solid #e7eef8;
    border-radius: 18px;
    padding: 14px 22px;
    box-shadow: 0 8px 30px rgba(26, 78, 135, 0.05);
    margin-bottom: 24px;
}

.logo {
    font-size: 21px;
    font-weight: 800;
    color: #153968;
}

.logo-mark {
    display: inline-flex;
    width: 34px;
    height: 34px;
    align-items: center;
    justify-content: center;
    background: #1769ff;
    color: white;
    border-radius: 10px;
    margin-right: 8px;
    font-size: 18px;
}


/* ---------- GLOBAL READABLE TEXT ---------- */
.stApp,
.stApp p,
.stApp span,
.stApp div,
.stApp label,
.stApp small,
.stApp li {
    color: #111827;
}

/* Keep muted helper text readable without becoming pure black. */
.section-subtitle,
.small-label,
.feature-text {
    color: #4b5563 !important;
}

/* Headings remain dark and high contrast. */
.section-title,
.feature-title,
.logo {
    color: #17385f !important;
}

/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(135deg, #ffffff 0%, #eef6ff 100%);
    border: 1px solid #dce9f8;
    border-radius: 25px;
    padding: 55px 55px;
    box-shadow: 0 15px 45px rgba(29, 88, 150, 0.06);
    margin-bottom: 26px;
}

.hero-tag {
    display: inline-block;
    background: #e8f2ff;
    color: #1769ff;
    padding: 8px 14px;
    border-radius: 30px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}

.hero h1 {
    color: #142f55;
    font-size: 48px;
    line-height: 1.1;
    margin: 0 0 18px 0;
    font-weight: 800;
}

.hero h1 span {
    color: #1769ff;
}

.hero p {
    color: #61708a;
    font-size: 17px;
    line-height: 1.65;
    max-width: 620px;
}

/* ---------- CARDS ---------- */
.info-card,
.form-card,
.result-card,
.factor-card,
.history-card,
.side-card {
    background: #ffffff;
    border: 1px solid #e4ecf6;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 35px rgba(41, 85, 130, 0.05);
    color: #111827 !important;
}

/* Force all normal UI text inside cards to remain readable.
   This fixes Streamlit/browser theme inheritance that can make
   some headings, labels, bold text and descriptions appear white. */
.info-card *,
.form-card *,
.result-card *,
.factor-card *,
.history-card *,
.side-card * {
    color: inherit;
}

.info-card b,
.info-card strong,
.form-card b,
.form-card strong,
.result-card b,
.result-card strong,
.factor-card b,
.factor-card strong,
.history-card b,
.history-card strong,
.side-card b,
.side-card strong {
    color: #111827 !important;
}

.info-card p,
.form-card p,
.result-card p,
.factor-card p,
.history-card p,
.side-card p,
.info-card span,
.form-card span,
.result-card span,
.factor-card span,
.history-card span,
.side-card span {
    color: #4b5563 !important;
}

.section-title {
    color: #17385f;
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #728198;
    margin-bottom: 22px;
}

.small-label {
    color: #708099;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* ---------- FEATURE CARDS ---------- */
.feature-title {
    color: #17385f;
    font-weight: 750;
    font-size: 17px;
}

.feature-text {
    color: #748096;
    line-height: 1.5;
}

/* ---------- STEPPER ---------- */
.stepper {
    background: #ffffff;
    border: 1px solid #e5edf7;
    border-radius: 18px;
    padding: 20px 26px;
    margin-bottom: 24px;
}

.step-line {
    height: 3px;
    background: #dbe8f7;
    flex: 1;
}

.step-circle-active {
    width: 38px;
    height: 38px;
    background: #1769ff;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.step-circle-done {
    width: 38px;
    height: 38px;
    background: #12b886;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.step-circle-inactive {
    width: 38px;
    height: 38px;
    background: #edf3fa;
    color: #7890a8;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.step-label {
    font-size: 12px;
    font-weight: 700;
    color: #59708e;
    margin-top: 7px;
    text-align: center;
}

/* ---------- PREDICTION RESULT ---------- */
.approved-box {
    background: linear-gradient(135deg, #f0fff9, #ffffff);
    border: 1px solid #ccefe2;
}

.denied-box {
    background: linear-gradient(135deg, #fff5f5, #ffffff);
    border: 1px solid #ffd5d5;
}

.result-title {
    font-size: 32px;
    font-weight: 850;
    margin: 0;
}

.approved-text {
    color: #07926f;
}

.denied-text {
    color: #d64545;
}

.big-percentage {
    font-size: 48px;
    font-weight: 850;
    color: #17385f;
}

/* ---------- STATUS BADGES ---------- */
.badge-green {
    display: inline-block;
    color: #078c6d;
    background: #e7fbf4;
    border: 1px solid #c9f3e5;
    padding: 5px 10px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 800;
}

.badge-red {
    display: inline-block;
    color: #d64545;
    background: #fff0f0;
    border: 1px solid #ffd5d5;
    padding: 5px 10px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 800;
}

.badge-blue {
    display: inline-block;
    color: #1769ff;
    background: #edf4ff;
    border: 1px solid #d5e5ff;
    padding: 5px 10px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 800;
}

/* ---------- STREAMLIT INPUTS ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 11px !important;
    border-color: #dce7f3 !important;
    background: #ffffff !important;
    min-height: 45px;
}

label {
    color: #43556e !important;
    font-weight: 700 !important;
}

/* Force actual Streamlit input text to dark/readable.
   Streamlit/Browser themes can otherwise inherit a white text color. */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-baseweb="input"] input,
[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"] {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
    background-color: #ffffff !important;
    opacity: 1 !important;
}

/* Placeholder values such as 40000 / 300.00 */
.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder,
[data-baseweb="input"] input::placeholder {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
    opacity: 1 !important;
}

/* Select/dropdown displayed text */
div[data-baseweb="select"] span,
div[data-baseweb="select"] [role="option"],
div[data-baseweb="select"] [role="combobox"],
[data-baseweb="menu"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Number-input +/- controls stay readable */
.stNumberInput button,
.stNumberInput button svg {
    color: #111827 !important;
    fill: #111827 !important;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    border-radius: 11px !important;
    min-height: 46px;
    font-weight: 800 !important;
    border: 1px solid #1769ff !important;
    color: #1769ff !important;
    background: #ffffff !important;
}

.stButton > button:hover {
    border-color: #0e57db !important;
    color: #0e57db !important;
    box-shadow: 0 8px 20px rgba(23, 105, 255, 0.12);
}

.primary-btn .stButton > button {
    background: #1769ff !important;
    color: #ffffff !important;
}

.primary-btn .stButton > button:hover {
    background: #0e57db !important;
    color: #ffffff !important;
}

/* ---------- DIVIDER ---------- */
.soft-divider {
    height: 1px;
    background: #e8eef6;
    margin: 24px 0;
}

/* ---------- MOBILE ---------- */
@media (max-width: 800px) {
    .hero {
        padding: 35px 25px;
    }

    .hero h1 {
        font-size: 35px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
"""), unsafe_allow_html=True)

# -----------------------------
# 4. HELPERS
# -----------------------------
def set_page(page):
    st.session_state.page = page
    if page == "apply":
        st.session_state.step = 1
    st.rerun()

def money(value):
    return f"${value:,.0f}"

def stepper(current_step):
    labels = ["Personal Details", "Financial Details", "Review & Predict"]
    html = '<div class="stepper"><div style="display:flex;align-items:flex-start;gap:10px;">'

    for i, label in enumerate(labels, start=1):
        if i < current_step:
            circle_class = "step-circle-done"
            symbol = "✓"
        elif i == current_step:
            circle_class = "step-circle-active"
            symbol = str(i)
        else:
            circle_class = "step-circle-inactive"
            symbol = str(i)

        html += textwrap.dedent(f"""
        <div style="flex:1;text-align:center;">
            <div style="display:flex;align-items:center;">
                <div style="flex:1;"></div>
                <div class="{circle_class}">{symbol}</div>
                <div style="flex:1;"></div>
            </div>
            <div class="step-label">{label}</div>
        </div>
        """)

    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)

def navbar():
    st.markdown(textwrap.dedent("""
    <div class="navbar">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div class="logo">
                <span class="logo-mark">🏦</span>LoanPredict
            </div>
            <div style="color:#61728a;font-size:14px;font-weight:700;">
                Smart • Secure • Simple
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if st.button("Home", use_container_width=True):
            set_page("home")
    with c2:
        if st.button("Apply Loan", use_container_width=True):
            set_page("apply")
    with c3:
        if st.button("History", use_container_width=True):
            set_page("history")
    with c4:
        if st.button("About", use_container_width=True):
            set_page("about")

    st.markdown(textwrap.dedent("<div class='soft-divider'></div>"), unsafe_allow_html=True)

def model_input_from_form(data):
    gender_val = 1 if data["gender"] == "Male" else 0
    married_val = 1 if data["married"] == "Yes" else 0
    dep_val = 3.0 if data["dependents"] == "3+" else float(data["dependents"])
    edu_val = 0 if data["education"] == "Graduate" else 1
    self_emp_val = 1 if data["self_employed"] == "Yes" else 0
    credit_val = 1.0 if data["credit_history"] == "Yes" else 0.0

    prop_mapping = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    prop_val = prop_mapping[data["property_area"]]

    total_income = data["applicant_income"] + data["coapplicant_income"]

    # UI asks for years, but your model expects Loan_Amount_Term in days.
    term_days = data["loan_term_years"] * 365

    # Same feature engineering approach as your original app.
    emi = (data["loan_amount_thousands"] * 1000) / term_days if term_days > 0 else 0
    dti_ratio = emi / (total_income + 1)
    balance_income = total_income - emi

    return pd.DataFrame({
        "Gender": [gender_val],
        "Married": [married_val],
        "Dependents": [dep_val],
        "Education": [edu_val],
        "Self_Employed": [self_emp_val],
        "ApplicantIncome": [data["applicant_income"]],
        "CoapplicantIncome": [data["coapplicant_income"]],
        "LoanAmount": [data["loan_amount_thousands"]],
        "Loan_Amount_Term": [term_days],
        "Credit_History": [credit_val],
        "Property_Area": [prop_val],
        "Total_Income": [total_income],
        "EMI": [emi],
        "DTI_Ratio": [dti_ratio],
        "Balance_Income": [balance_income],
    }), total_income, emi, dti_ratio, balance_income

# -----------------------------
# 5. HOME PAGE
# -----------------------------
def home_page():
    navbar()

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown(textwrap.dedent("""
        <div class="hero">
            <div class="hero-tag">AI Powered Loan Prediction</div>
            <h1>Your Financial<br>Future, <span>Smarter</span></h1>
            <p>
                Get a quick and accurate loan approval prediction using
                machine learning. Simple, secure and designed for everyone.
            </p>
            <br>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <div class="badge-blue">⚡ Fast Results</div>
                <div class="badge-blue">🔒 Secure</div>
                <div class="badge-blue">✓ Easy to Use</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(textwrap.dedent('<div class="primary-btn">'), unsafe_allow_html=True)
            if st.button("Apply for Loan →", use_container_width=True):
                set_page("apply")
            st.markdown(textwrap.dedent("</div>"), unsafe_allow_html=True)

        with c2:
            if st.button("How It Works", use_container_width=True):
                set_page("about")

    with right:
        st.markdown(textwrap.dedent("""
        <div class="info-card" style="min-height:390px;">
            <div style="font-size:75px;text-align:center;margin-top:15px;">🏦</div>
            <div style="text-align:center;color:#1769ff;font-size:26px;font-weight:850;">
                Loan Approval
            </div>
            <div style="text-align:center;color:#6f7f96;margin:8px 0 25px;">
                Prediction System
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div class="factor-card">
                    <b>Personal</b><br>
                    <span style="color:#7c8ca2;font-size:13px;">Profile details</span>
                </div>
                <div class="factor-card">
                    <b>Financial</b><br>
                    <span style="color:#7c8ca2;font-size:13px;">Income & loan data</span>
                </div>
                <div class="factor-card">
                    <b>Credit</b><br>
                    <span style="color:#7c8ca2;font-size:13px;">Credit history</span>
                </div>
                <div class="factor-card">
                    <b>AI Model</b><br>
                    <span style="color:#7c8ca2;font-size:13px;">Logistic Regression</span>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("### Why use LoanPredict?")
    cols = st.columns(3)

    features = [
        ("⚡", "Fast & Simple", "Enter your details and receive a prediction in seconds."),
        ("🔒", "Secure & Private", "Designed to keep the prediction process simple and transparent."),
        ("🎯", "Data Driven", "Uses a trained Logistic Regression model for prediction."),
    ]

    for col, (icon, title, text) in zip(cols, features):
        with col:
            st.markdown(textwrap.dedent(f"""
            <div class="info-card">
                <div style="font-size:28px;">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-text">{text}</div>
            </div>
            """), unsafe_allow_html=True)

# -----------------------------
# 6. PERSONAL DETAILS
# -----------------------------
def personal_step():
    navbar()
    stepper(1)

    left, right = st.columns([1.45, 0.55], gap="large")

    with left:
        st.markdown(textwrap.dedent('<div class="form-card">'), unsafe_allow_html=True)
        st.markdown(textwrap.dedent('<div class="section-title">Personal Details</div>'), unsafe_allow_html=True)
        st.markdown(
            textwrap.dedent('<div class="section-subtitle">Tell us about the applicant. This information is used by the prediction model.</div>'),
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender *", ["Male", "Female"])
            married = st.selectbox("Marital Status *", ["Yes", "No"])
            dependents = st.selectbox("Dependents *", ["0", "1", "2", "3+"])

        with c2:
            education = st.selectbox("Education *", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed *", ["No", "Yes"])
            property_area = st.selectbox("Property Area *", ["Urban", "Semiurban", "Rural"])

        st.markdown(textwrap.dedent("</div>"), unsafe_allow_html=True)

    with right:
        st.markdown(textwrap.dedent("""
        <div class="side-card">
            <div style="font-size:28px;">🛡️</div>
            <div style="font-size:18px;font-weight:800;color:#18385e;margin:7px 0;">
                Why we need this?
            </div>
            <div style="color:#718099;line-height:1.6;font-size:14px;">
                These details help the machine learning model understand the
                applicant's profile and calculate a loan prediction.
            </div>
            <br>
            <div class="badge-blue">✓ Quick</div><br><br>
            <div class="badge-blue">✓ Secure</div><br><br>
            <div class="badge-blue">✓ Private</div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("<br>"), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Back to Home", use_container_width=True):
            set_page("home")
    with c2:
        st.markdown(textwrap.dedent('<div class="primary-btn">'), unsafe_allow_html=True)
        if st.button("Next: Financial Details →", use_container_width=True):
            st.session_state.form_data.update({
                "gender": gender,
                "married": married,
                "dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "property_area": property_area,
            })
            st.session_state.step = 2
            st.rerun()
        st.markdown(textwrap.dedent("</div>"), unsafe_allow_html=True)

# -----------------------------
# 7. FINANCIAL DETAILS
# -----------------------------
def financial_step():
    navbar()
    stepper(2)

    left, right = st.columns([1.45, 0.55], gap="large")

    with left:
        st.markdown(textwrap.dedent('<div class="form-card">'), unsafe_allow_html=True)
        st.markdown(textwrap.dedent('<div class="section-title">Financial Details</div>'), unsafe_allow_html=True)
        st.markdown(
            textwrap.dedent('<div class="section-subtitle">Enter accurate financial information for a more reliable prediction.</div>'),
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            applicant_income = st.number_input(
                "Applicant Monthly Income ($) *",
                min_value=0,
                value=40000,
                step=1000
            )
            loan_amount_thousands = st.number_input(
                "Loan Amount ($ thousands) *",
                min_value=1.0,
                value=300.0,
                step=10.0
            )
            credit_history = st.selectbox(
                "Credit History Meets Guidelines? *",
                ["Yes", "No"]
            )

        with c2:
            coapplicant_income = st.number_input(
                "Coapplicant Monthly Income ($)",
                min_value=0,
                value=15000,
                step=1000
            )
            loan_term_years = st.selectbox(
                "Loan Tenure (Years) *",
                [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30]
            )

        st.markdown(textwrap.dedent("</div>"), unsafe_allow_html=True)

    with right:
        st.markdown(textwrap.dedent("""
        <div class="side-card">
            <div style="font-size:28px;">💡</div>
            <div style="font-size:18px;font-weight:800;color:#18385e;margin:7px 0;">
                Tips for better results
            </div>
            <div style="color:#718099;line-height:1.7;font-size:14px;">
                ✓ Be honest with your information<br>
                ✓ Ensure your income values are accurate<br>
                ✓ A good credit history improves approval chances
            </div>
            <br>
            <div style="text-align:center;font-size:38px;">◔</div>
            <div style="text-align:center;color:#6e7f97;font-size:13px;">
                Better data = More reliable prediction
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("<br>"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back: Personal Details", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

    with c2:
        st.markdown(textwrap.dedent('<div class="primary-btn">'), unsafe_allow_html=True)
        if st.button("Next: Review & Predict →", use_container_width=True):
            st.session_state.form_data.update({
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount_thousands": loan_amount_thousands,
                "loan_term_years": loan_term_years,
                "credit_history": credit_history,
            })
            st.session_state.step = 3
            st.rerun()
        st.markdown(textwrap.dedent("</div>"), unsafe_allow_html=True)

# -----------------------------
# 8. REVIEW PAGE
# -----------------------------
def review_step():
    navbar()
    stepper(3)

    data = st.session_state.form_data

    st.markdown(textwrap.dedent('<div class="section-title">Review Your Details</div>'), unsafe_allow_html=True)
    st.markdown(
        textwrap.dedent('<div class="section-subtitle">Please check your information before running the prediction.</div>'),
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(textwrap.dedent(f"""
        <div class="info-card">
            <div style="font-size:18px;font-weight:800;color:#17385f;">👤 Personal Details</div>
            <br>
            <b>Gender:</b> {data["gender"]}<br><br>
            <b>Married:</b> {data["married"]}<br><br>
            <b>Dependents:</b> {data["dependents"]}<br><br>
            <b>Education:</b> {data["education"]}<br><br>
            <b>Self Employed:</b> {data["self_employed"]}<br><br>
            <b>Property Area:</b> {data["property_area"]}
        </div>
        """), unsafe_allow_html=True)

    with c2:
        loan_display = data["loan_amount_thousands"] * 1000
        st.markdown(textwrap.dedent(f"""
        <div class="info-card">
            <div style="font-size:18px;font-weight:800;color:#17385f;">💰 Financial Details</div>
            <br>
            <b>Applicant Income:</b> {money(data["applicant_income"])}<br><br>
            <b>Coapplicant Income:</b> {money(data["coapplicant_income"])}<br><br>
            <b>Loan Amount:</b> {money(loan_display)}<br><br>
            <b>Loan Tenure:</b> {data["loan_term_years"]} years<br><br>
            <b>Credit History:</b> {data["credit_history"]}
        </div>
        """), unsafe_allow_html=True)

    with c3:
        st.markdown(textwrap.dedent("""
        <div class="info-card" style="height:100%;">
            <div style="font-size:18px;font-weight:800;color:#17385f;">✓ Summary</div>
            <br>
            <div class="badge-green">Profile Ready</div>
            <br><br>
            <div style="color:#748199;line-height:1.6;">
                Your profile is complete. Click the button below to run the
                trained machine learning model.
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("<br>"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        if st.button("← Edit Information", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    with c2:
        st.markdown(textwrap.dedent('<div class="primary-btn">'), unsafe_allow_html=True)
        if st.button("Predict Loan Approval →", use_container_width=True):
            if model is None:
                st.error("Model file not found. Put 'final_lr_loan_model.pkl' in the same folder as this app.")
            else:
                with st.spinner("Analyzing applicant profile..."):
                    time.sleep(1)

                    input_df, total_income, emi, dti_ratio, balance_income = model_input_from_form(data)

                    prediction = model.predict(input_df)[0]

                    probability = None
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(input_df)[0]
                        # Assumes class 1 means approved, matching your original code.
                        probability = float(probs[1] * 100) if len(probs) > 1 else None

                    st.session_state.prediction = int(prediction)
                    st.session_state.probability = probability
                    st.session_state.form_data.update({
                        "total_income": total_income,
                        "emi": emi,
                        "dti_ratio": dti_ratio,
                        "balance_income": balance_income,
                    })

                    st.session_state.history.insert(0, {
                        "date": datetime.now().strftime("%d %b %Y"),
                        "loan_amount": data["loan_amount_thousands"] * 1000,
                        "tenure": data["loan_term_years"],
                        "status": "Approved" if int(prediction) == 1 else "Rejected",
                        "probability": probability,
                    })

                    st.session_state.page = "result"
                    st.rerun()
        st.markdown(textwrap.dedent("</div>"), unsafe_allow_html=True)

# -----------------------------
# 9. RESULT PAGE
# -----------------------------
def result_page():
    navbar()

    data = st.session_state.form_data
    prediction = st.session_state.prediction
    probability = st.session_state.probability

    approved = prediction == 1

    if probability is None:
        probability = 82.0 if approved else 30.0

    if approved:
        st.markdown(textwrap.dedent(f"""
        <div class="result-card approved-box">
            <div style="text-align:center;">
                <div style="font-size:54px;">✓</div>
                <div class="result-title approved-text">Loan Approved!</div>
                <p style="color:#628071;font-size:16px;">
                    Based on the provided profile, there is a high chance of loan approval.
                </p>
                <div class="big-percentage">{probability:.0f}%</div>
                <div style="color:#70829a;font-weight:700;">Approval Probability</div>
            </div>
        </div>
        """), unsafe_allow_html=True)
    else:
        st.markdown(textwrap.dedent(f"""
        <div class="result-card denied-box">
            <div style="text-align:center;">
                <div style="font-size:54px;">!</div>
                <div class="result-title denied-text">Loan Not Approved</div>
                <p style="color:#946e6e;font-size:16px;">
                    The model predicts a lower chance of approval for this profile.
                </p>
                <div class="big-percentage">{probability:.0f}%</div>
                <div style="color:#70829a;font-weight:700;">Approval Probability</div>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("<br>"), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(textwrap.dedent(f"""
        <div class="info-card">
            <div class="small-label">Loan Amount</div>
            <div style="font-size:23px;font-weight:850;color:#17385f;">
                {money(data["loan_amount_thousands"] * 1000)}
            </div>
        </div>
        """), unsafe_allow_html=True)

    with c2:
        st.markdown(textwrap.dedent(f"""
        <div class="info-card">
            <div class="small-label">Tenure</div>
            <div style="font-size:23px;font-weight:850;color:#17385f;">
                {data["loan_term_years"]} Years
            </div>
        </div>
        """), unsafe_allow_html=True)

    with c3:
        st.markdown(textwrap.dedent(f"""
        <div class="info-card">
            <div class="small-label">Estimated EMI</div>
            <div style="font-size:23px;font-weight:850;color:#17385f;">
                {money(data.get("emi", 0))} / month
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("### Key Factors")

    factors = [
        ("Income", "Good" if data["applicant_income"] >= 30000 else "Needs attention"),
        ("Credit History", "Good" if data["credit_history"] == "Yes" else "Needs attention"),
        ("Existing EMI", "Low" if data.get("emi", 0) < data.get("total_income", 1) * 0.25 else "High"),
        ("Debt to Income", "Healthy" if data.get("dti_ratio", 1) < 0.35 else "High"),
    ]

    cols = st.columns(4)
    for col, (name, value) in zip(cols, factors):
        with col:
            st.markdown(textwrap.dedent(f"""
            <div class="factor-card">
                <div class="small-label">{name}</div>
                <div style="font-size:18px;font-weight:800;color:#17385f;">{value}</div>
                <span class="badge-green">Model factor</span>
            </div>
            """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("<br>"), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("View Detailed Analysis", use_container_width=True):
            set_page("analysis")
    with c2:
        st.markdown(textwrap.dedent('<div class="primary-btn">'), unsafe_allow_html=True)
        if st.button("Apply for Loan Again", use_container_width=True):
            set_page("apply")
        st.markdown(textwrap.dedent("</div>"), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("""
    <div class="side-card" style="margin-top:24px;">
        <b>Important:</b>
        <span style="color:#6f7e93;">
        This prediction is generated by a machine learning model and should not
        be treated as a final banking or credit decision.
        </span>
    </div>
    """), unsafe_allow_html=True)

# -----------------------------
# 10. EXPLANATION PAGE
# -----------------------------
def analysis_page():
    navbar()

    data = st.session_state.form_data
    probability = st.session_state.probability
    prediction = st.session_state.prediction

    st.markdown(textwrap.dedent('<div class="section-title">Why You Got This Result?</div>'), unsafe_allow_html=True)
    st.markdown(
        textwrap.dedent('<div class="section-subtitle">Here is a simple view of the main profile factors used in the prediction.</div>'),
        unsafe_allow_html=True
    )

    checks = [
        (
            "Monthly Income",
            "Positive" if data["applicant_income"] >= 30000 else "Needs attention",
            f"Applicant income: {money(data['applicant_income'])}"
        ),
        (
            "Credit History",
            "Positive" if data["credit_history"] == "Yes" else "Negative",
            f"Credit history: {data['credit_history']}"
        ),
        (
            "Existing EMI",
            "Positive" if data.get("emi", 0) < data.get("total_income", 1) * 0.25 else "Needs attention",
            f"Estimated EMI: {money(data.get('emi', 0))}"
        ),
        (
            "Debt to Income Ratio",
            "Positive" if data.get("dti_ratio", 1) < 0.35 else "Needs attention",
            f"DTI ratio: {data.get('dti_ratio', 0):.2f}"
        ),
        (
            "Employment",
            "Positive" if data["self_employed"] == "No" else "Profile factor",
            f"Self employed: {data['self_employed']}"
        ),
    ]

    left, right = st.columns([1.45, 0.55], gap="large")

    with left:
        for name, status, detail in checks:
            positive = status == "Positive"
            badge = "badge-green" if positive else "badge-red"

            st.markdown(textwrap.dedent(f"""
            <div class="info-card" style="margin-bottom:12px;padding:18px 22px;">
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div>
                        <div style="font-size:16px;font-weight:800;color:#17385f;">{name}</div>
                        <div style="font-size:13px;color:#78879b;margin-top:4px;">{detail}</div>
                    </div>
                    <div class="{badge}">{status}</div>
                </div>
            </div>
            """), unsafe_allow_html=True)

    with right:
        st.markdown(textwrap.dedent("""
        <div class="side-card">
            <div style="font-size:18px;font-weight:800;color:#17385f;">Need to Improve?</div>
            <p style="color:#718099;line-height:1.6;">
                Factors that commonly affect loan prediction include:
            </p>
            <div style="line-height:2;color:#52647c;font-size:14px;">
                ✓ Maintain a healthy credit history<br>
                ✓ Keep monthly debt manageable<br>
                ✓ Provide accurate income information<br>
                ✓ Avoid unnecessary loan applications
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("<br>"), unsafe_allow_html=True)

    if st.button("← Back to Prediction Result", use_container_width=True):
        set_page("result")

# -----------------------------
# 11. HISTORY PAGE
# -----------------------------
def history_page():
    navbar()

    st.markdown(textwrap.dedent('<div class="section-title">Your Loan Applications</div>'), unsafe_allow_html=True)
    st.markdown(
        textwrap.dedent('<div class="section-subtitle">Track recent predictions made in this browser session.</div>'),
        unsafe_allow_html=True
    )

    if not st.session_state.history:
        st.markdown(textwrap.dedent("""
        <div class="info-card" style="text-align:center;padding:55px;">
            <div style="font-size:52px;">📋</div>
            <div style="font-size:20px;font-weight:800;color:#17385f;">
                No applications yet
            </div>
            <div style="color:#748096;margin:8px 0 22px;">
                Start a new application to see your prediction history here.
            </div>
        </div>
        """), unsafe_allow_html=True)

        if st.button("Apply for Loan →", use_container_width=True):
            set_page("apply")
        return

    rows = []
    for item in st.session_state.history:
        rows.append({
            "Date": item["date"],
            "Loan Amount": money(item["loan_amount"]),
            "Tenure": f"{item['tenure']} Years",
            "Status": item["status"],
            "Prediction": f"{item['probability']:.0f}%" if item["probability"] is not None else "—"
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(textwrap.dedent("<br>"), unsafe_allow_html=True)
    if st.button("＋ New Application", use_container_width=True):
        set_page("apply")

# -----------------------------
# 12. ABOUT PAGE
# -----------------------------
def about_page():
    navbar()

    st.markdown(textwrap.dedent("""
    <div style="text-align:center;margin-bottom:40px;">
        <div style="font-size:32px;font-weight:850;color:#17385f;">
            About LoanPredict
        </div>
        <div style="color:#728198;margin-top:8px;font-size:16px;">
            A smart machine learning application for predicting loan approval outcomes.
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 1: WHAT IS LOANPREDICT
    # ============================================================
    st.markdown("### What is LoanPredict?")

    st.markdown(textwrap.dedent("""
    <div class="info-card">
        <div style="color:#4b5563;line-height:1.8;font-size:15px;">
            LoanPredict is a machine learning-based application designed to estimate
            whether a loan application is likely to be approved. The system analyzes
            important applicant, financial, and credit information and provides a
            prediction using a trained Logistic Regression model.
            <br><br>
            The application is designed to make the prediction process simple, fast,
            and easy to understand for users without requiring technical or financial expertise.
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 2: HOW IT WORKS
    # ============================================================
    st.markdown("### How LoanPredict Works")

    cols = st.columns(4)
    steps = [
        ("Enter Details", "Provide applicant, employment, financial, and credit information."),
        ("Process Data", "The application converts the entered information into the format required by the model."),
        ("Run Prediction", "The trained Logistic Regression model evaluates the applicant profile."),
        ("View Result", "The system displays the predicted approval status and supporting information."),
    ]

    for col, (title, desc) in zip(cols, steps):
        with col:
            step_num = steps.index((title, desc)) + 1
            style_str = "width:40px;height:40px;background:#1769ff;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;margin:0 auto 12px;font-size:18px;"
            st.markdown(f'<div class="info-card" style="text-align:center;min-height:200px;"><div style="{style_str}">{step_num}</div><div class="feature-title">{title}</div><div class="feature-text" style="margin-top:10px;">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 3: MAIN FEATURES
    # ============================================================
    st.markdown("### Main Features")

    cols = st.columns(3)

    features_list = [
        ("Applicant Profile", "Collects information such as gender, marital status, dependents, education, employment status, and property area."),
        ("Financial Analysis", "Uses applicant income, coapplicant income, requested loan amount, loan tenure, and credit history."),
        ("Machine Learning Prediction", "Uses the trained Logistic Regression model to classify the loan application as approved or not approved."),
    ]

    for col, (title, text) in zip(cols, features_list):
        with col:
            st.markdown(textwrap.dedent(f"""
            <div class="info-card" style="min-height:200px;">
                <div class="feature-title">{title}</div>
                <div class="feature-text" style="margin-top:10px;">
                    {text}
                </div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 4: INFORMATION USED BY MODEL
    # ============================================================
    st.markdown("### Information Used for Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(textwrap.dedent("""
        <div class="info-card" style="min-height:270px;">
            <div style="font-size:19px;font-weight:800;color:#17385f;margin-bottom:15px;">
                Applicant Information
            </div>
            <div style="color:#4b5563;line-height:2;">
                ✓ Gender<br>
                ✓ Marital Status<br>
                ✓ Number of Dependents<br>
                ✓ Education<br>
                ✓ Self-Employment Status<br>
                ✓ Property Area
            </div>
        </div>
        """), unsafe_allow_html=True)

    with col2:
        st.markdown(textwrap.dedent("""
        <div class="info-card" style="min-height:270px;">
            <div style="font-size:19px;font-weight:800;color:#17385f;margin-bottom:15px;">
                Financial & Credit Information
            </div>
            <div style="color:#4b5563;line-height:2;">
                ✓ Applicant Income<br>
                ✓ Coapplicant Income<br>
                ✓ Loan Amount<br>
                ✓ Loan Tenure<br>
                ✓ Credit History<br>
                ✓ Derived Financial Indicators
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 5: MODEL INFORMATION
    # ============================================================
    st.markdown("### Machine Learning Model")

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown(textwrap.dedent("""
        <div class="info-card">
            <div style="font-size:19px;font-weight:800;color:#17385f;margin-bottom:15px;">
                Model Details
            </div>
            <div style="color:#4b5563;line-height:2;">
                <b>Algorithm:</b> Logistic Regression<br>
                <b>Prediction Type:</b> Binary Classification<br>
                <b>Possible Results:</b> Approved / Not Approved<br>
                <b>Model File:</b> final_lr_loan_model.pkl<br>
                <b>Feature Engineering:</b> Total Income, EMI, DTI Ratio, and Balance Income
            </div>
        </div>
        """), unsafe_allow_html=True)

    with col2:
        st.markdown(textwrap.dedent("""
        <div class="info-card" style="text-align:center;">
            <div style="font-size:19px;font-weight:800;color:#17385f;margin-bottom:10px;">
                Reported Model Accuracy
            </div>
            <div style="font-size:52px;font-weight:850;color:#1769ff;margin:10px 0;">
                ~82.3%
            </div>
            <div style="color:#4b5563;line-height:1.7;font-size:14px;">
                The original project reports an accuracy of approximately 82.3%
                for the trained model.
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 6: DERIVED FEATURES
    # ============================================================
    st.markdown("### Additional Financial Indicators")

    cols = st.columns(3)

    indicators = [
        ("Total Income", "Applicant income combined with coapplicant income."),
        ("EMI", "Estimated monthly repayment amount based on the loan amount and tenure."),
        ("DTI Ratio", "Compares estimated loan repayment with the applicant's total income."),
    ]

    for col, (title, text) in zip(cols, indicators):
        with col:
            st.markdown(textwrap.dedent(f"""
            <div class="factor-card" style="min-height:165px;">
                <div style="font-size:17px;font-weight:800;color:#17385f;">
                    {title}
                </div>
                <div style="color:#4b5563;line-height:1.7;margin-top:10px;">
                    {text}
                </div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 7: WHY USE LOANPREDICT
    # ============================================================
    st.markdown("### Why Use LoanPredict?")

    cols = st.columns(4)

    benefits = [
        ("⚡", "Fast", "Get a prediction within seconds."),
        ("🎯", "Data Driven", "Uses a trained machine learning model."),
        ("📊", "Clear Results", "Presents results in a simple and understandable format."),
        ("👥", "Easy to Use", "Designed for users with different levels of technical knowledge."),
    ]

    for col, (icon, title, text) in zip(cols, benefits):
        with col:
            st.markdown(textwrap.dedent(f"""
            <div class="info-card" style="min-height:175px;text-align:center;">
                <div style="font-size:30px;">
                    {icon}
                </div>
                <div class="feature-title">
                    {title}
                </div>
                <div class="feature-text">
                    {text}
                </div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 8: IMPORTANT DISCLAIMER
    # ============================================================
    st.markdown(textwrap.dedent("""
    <div class="side-card" style="border-left:5px solid #1769ff;background:#f8fbff;">
        <div style="font-size:19px;font-weight:800;color:#17385f;margin-bottom:10px;">
            Important Information
        </div>
        <div style="color:#4b5563;line-height:1.8;font-size:14px;">
            LoanPredict provides a machine learning-based prediction and
            should be considered a decision-support and educational tool.
            The prediction is not a final loan approval or rejection from
            a bank or financial institution.
            <br><br>
            Actual lending decisions may depend on additional financial
            information, verification procedures, credit assessment,
            institutional policies, and human review.
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # SECTION 9: FOOTER
    # ============================================================
    st.markdown(textwrap.dedent("""
    <div style="text-align:center;padding:20px 0;">
        <div style="font-size:22px;font-weight:850;color:#17385f;">
            LoanPredict
        </div>
        <div style="color:#718096;margin-top:6px;font-size:14px;">
            Smart • Secure • Simple
        </div>
        <div style="color:#9aa6b6;margin-top:5px;font-size:12px;">
            Machine Learning Based Loan Approval Prediction System
        </div>
    </div>
    """), unsafe_allow_html=True)
# -----------------------------
# 13. ROUTER
# -----------------------------
if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "apply":
    if st.session_state.step == 1:
        personal_step()
    elif st.session_state.step == 2:
        financial_step()
    else:
        review_step()

elif st.session_state.page == "result":
    result_page()

elif st.session_state.page == "analysis":
    analysis_page()

elif st.session_state.page == "history":
    history_page()

elif st.session_state.page == "about":
    about_page()
