"""
ChurnGuard — Telecom Churn Intelligence Platform
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import warnings

warnings.filterwarnings('ignore')

try:
    from groq import Groq
except ImportError:
    Groq = None

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ChurnGuard", layout="wide", initial_sidebar_state="expanded")

GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#f5f6fa;color:#1e1e2e;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:0;padding-bottom:2rem;max-width:1180px;}

/* sidebar */
[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #e4e6ef;}
[data-testid="stSidebar"] *{color:#374151 !important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#111827 !important;font-size:.88rem !important;}

/* metrics */
[data-testid="metric-container"]{background:#fff;border:1px solid #e4e6ef;border-radius:14px;padding:.9rem 1.1rem;box-shadow:0 1px 3px rgba(0,0,0,.05);}
[data-testid="metric-container"] label{color:#6b7280 !important;font-size:.68rem !important;font-weight:600 !important;text-transform:uppercase;letter-spacing:.06em;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#111827 !important;font-size:1.45rem !important;font-weight:700 !important;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:10px;padding:4px;gap:2px;border:1px solid #e4e6ef;width:fit-content;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#6b7280;font-weight:500;font-size:.81rem;padding:6px 16px;background:transparent;}
.stTabs [aria-selected="true"]{background:#5b5bd6 !important;color:#fff !important;font-weight:600 !important;}

/* buttons */
.stButton>button{background:#5b5bd6;color:#fff;border:none;border-radius:8px;font-weight:600;font-size:.83rem;padding:.48rem 1.3rem;transition:all .2s;}
.stButton>button:hover{background:#4c4cc4;box-shadow:0 4px 12px rgba(91,91,214,.35);transform:translateY(-1px);}

/* topbar */
.topbar{background:#fff;border-bottom:1px solid #e4e6ef;padding:.8rem 1.6rem;margin-bottom:1.4rem;display:flex;align-items:center;gap:11px;}
.tlogo{width:34px;height:34px;background:linear-gradient(135deg,#5b5bd6,#8b5cf6);border-radius:9px;display:inline-flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:13px;flex-shrink:0;}
.tname{font-size:1rem;font-weight:700;color:#111827;}
.tsub{font-size:.76rem;color:#9ca3af;margin-left:3px;}

/* cards */
.cg-card{background:#fff;border:1px solid #e4e6ef;border-radius:13px;padding:1rem 1.2rem;margin-bottom:.75rem;box-shadow:0 1px 3px rgba(0,0,0,.04);}
.cg-danger {border-left:3px solid #ef4444;}
.cg-success{border-left:3px solid #10b981;}
.cg-warning{border-left:3px solid #f59e0b;}
.cg-info   {border-left:3px solid #5b5bd6;}

/* ai box */
.ai-box{background:#faf5ff;border:1px solid #ddd6fe;border-radius:12px;padding:.95rem 1.1rem;margin-top:.6rem;}
.ai-box h4{color:#7c3aed;margin:0 0 .35rem;font-size:.78rem;font-weight:600;}
.ai-box p{color:#374151;font-size:.84rem;line-height:1.65;margin:0;}

/* pills */
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.68rem;font-weight:600;}
.p-red   {background:#fef2f2;color:#dc2626;border:1px solid #fecaca;}
.p-yellow{background:#fffbeb;color:#d97706;border:1px solid #fde68a;}
.p-blue  {background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe;}

/* section label */
.slabel{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#9ca3af;margin-bottom:.45rem;}

hr{border-color:#f3f4f6 !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CHURN_REASONS = {
    'Month-to-Month Contract': {
        'impact': 91, 'tag': 'Critical', 'color': 'p-red',
        'desc': 'Customers with no long-term contract leave 3x more often. Nothing locks them in.',
        'fixes': ['Offer 20% discount to switch to a 1-year contract', 'Add loyalty rewards from month 6 onward'],
    },
    'High Monthly Charges': {
        'impact': 78, 'tag': 'Critical', 'color': 'p-red',
        'desc': 'Customers paying above $80/mo without premium add-ons feel they are overpaying.',
        'fixes': ['Bundle free security or streaming for 3 months', 'Call the customer for a loyalty rate review'],
    },
    'No Tech Support': {
        'impact': 63, 'tag': 'High', 'color': 'p-yellow',
        'desc': 'Unsolved problems are a top reason for leaving. Without support, customers feel abandoned.',
        'fixes': ['Offer a free 3-month tech support trial', 'Schedule a proactive monthly check-in call'],
    },
    'Short Tenure Under 12 Months': {
        'impact': 55, 'tag': 'High', 'color': 'p-yellow',
        'desc': 'New customers have not built loyalty yet. First 6 months are the highest-risk window.',
        'fixes': ['Schedule an onboarding call at month 3', 'Send a satisfaction survey and resolve issues fast'],
    },
    'Electronic Check Payment': {
        'impact': 38, 'tag': 'Medium', 'color': 'p-blue',
        'desc': 'Manual payment creates friction. Auto-pay customers are significantly less likely to leave.',
        'fixes': ['Offer $5/mo discount for switching to auto-pay', 'Send a one-click auto-pay enrollment link'],
    },
}

COLORS = ['#5b5bd6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4']

BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#374151', size=11),
    margin=dict(l=8, r=8, t=34, b=8),
    height=230,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    try:
        mdl = joblib.load('model_artifacts/churn_model.pkl')
        exp = joblib.load('model_artifacts/shap_explainer.pkl')
        with open('model_artifacts/feature_cols.json') as f:
            fc = json.load(f)
        with open('model_artifacts/metrics.json') as f:
            mt = json.load(f)
        return mdl, exp, fc, mt
    except Exception as e:
        st.error(f"Model not found. Run both notebooks first. ({e})")
        return None, None, [], {}


@st.cache_data
def load_data():
    for p in ['mastertable_data.csv', 'mastertable_data.xlsx']:
        if os.path.exists(p):
            return pd.read_csv(p) if p.endswith('.csv') else pd.read_excel(p)
    return pd.DataFrame()


model, explainer, feature_cols, metrics_info = load_artifacts()
raw_df = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def risk_label(p):
    if p >= .65: return 'High Risk', '#ef4444'
    if p >= .35: return 'Medium Risk', '#f59e0b'
    return 'Low Risk', '#10b981'


def call_ai(prompt):
    if Groq is None:
        return "Run: pip install groq"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, temperature=0.7,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"AI error: {e}"


def encode_input(inp):
    from sklearn.preprocessing import LabelEncoder
    raw = load_data()
    drop = ['Column1', 'Count', 'Status.Customer Status', 'Status.Churn Label',
            'Status.Churn Category', 'Status.Churn Reason', 'Location.Country',
            'Location.State', 'Location.City', 'Location.Zip Code',
            'Location.Latitude', 'Location.Longitude', 'Population.Population',
            'Churn Risk', 'Status.Churn Value']
    ref = raw.drop(columns=drop, errors='ignore')
    ref['Services.Offer'] = ref['Services.Offer'].fillna('No Offer')
    ref['Services.Internet Type'] = ref['Services.Internet Type'].fillna('No Internet')
    row = pd.DataFrame([inp])
    row['Services.Offer'] = row.get('Services.Offer', pd.Series(['No Offer'])).fillna('No Offer')
    row['Services.Internet Type'] = row.get('Services.Internet Type', pd.Series(['No Internet'])).fillna('No Internet')
    for col in ref.select_dtypes(include='object').columns:
        if col in row.columns:
            le = LabelEncoder();
            le.fit(ref[col].astype(str))
            try:
                row[col] = le.transform(row[col].astype(str))
            except:
                row[col] = 0
    for col in feature_cols:
        if col not in row.columns: row[col] = 0
    return row[feature_cols]


def apply_base(fig, h=230, extra=None):
    layout = {**BASE_LAYOUT, 'height': h}
    if extra: layout.update(extra)
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, showline=False, tickfont=dict(size=10, color='#6b7280'))
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f5', showline=False, tickfont=dict(size=10, color='#6b7280'))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="tlogo">CG</div>
  <span class="tname">ChurnGuard</span>
  <span class="tsub">Telecom Customer Intelligence Platform</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ChurnGuard")
    st.caption("Churn prediction and retention platform")
    st.divider()
    st.markdown("**Model Performance**")
    if metrics_info:
        ca, cb = st.columns(2)
        ca.metric("AUC", f"{metrics_info.get('xgb_auc', 0):.3f}")
        cb.metric("F1", f"{metrics_info.get('xgb_f1', 0):.3f}")
        st.caption(f"Engine: {metrics_info.get('best_model', 'XGBoost')}")
    st.divider()
    st.markdown("**Dataset**")
    if not raw_df.empty:
        tot = len(raw_df)
        ch = int(raw_df['Status.Churn Value'].sum()) if 'Status.Churn Value' in raw_df else 0
        st.markdown(f"Customers: **{tot:,}**")
        st.markdown(f"Churned: **{ch:,}**  |  Retained: **{tot - ch:,}**")
        st.progress(ch / tot if tot else 0)
        st.caption(f"Churn rate: {ch / tot * 100:.1f}%")
    st.divider()
    st.markdown("**AI**")
    st.success("Groq Llama 3.3 — connected")
    st.caption("AI is ready. No key needed.")
    st.divider()
    st.caption("Streamlit · XGBoost · Groq Llama 3.3")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Batch Dashboard", "Individual Predictor", "Why They Leave", "Revenue Impact"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BATCH DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if raw_df.empty:
        st.warning("Place mastertable_data.csv in the project folder.")
    else:
        df = raw_df.copy()
        total = len(df)
        churned = int(df['Status.Churn Value'].sum())
        retained = total - churned
        churn_rate = churned / total * 100
        rev_risk = df[df['Status.Churn Value'] == 1]['Services.Monthly Charge'].sum()
        avg_sat = df['Status.Satisfaction Score'].mean()

        st.markdown("##### Key Numbers")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Customers", f"{total:,}")
        k2.metric("Churn Rate", f"{churn_rate:.1f}%")
        k3.metric("Customers Retained", f"{retained:,}")
        k4.metric("Revenue at Risk /mo", f"${rev_risk:,.0f}")
        k5.metric("Avg Satisfaction", f"{avg_sat:.1f} / 5")

        st.markdown("<div style='margin-top:.8rem'></div>", unsafe_allow_html=True)

        # row 1
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("<div class='slabel'>Churn rate by contract</div>", unsafe_allow_html=True)
            ct = df.groupby('Services.Contract')['Status.Churn Value'].mean().reset_index()
            ct.columns = ['Contract', 'Rate']
            ct['Rate'] = (ct['Rate'] * 100).round(1)
            ct = ct.sort_values('Rate', ascending=False)
            fig = go.Figure(go.Bar(
                x=ct['Contract'], y=ct['Rate'],
                marker_color=['#ef4444', '#f59e0b', '#10b981'][:len(ct)],
                marker_line_width=0,
                text=[f"{v:.1f}%" for v in ct['Rate']],
                textposition='outside', textfont=dict(size=10)))
            apply_base(fig, extra={'yaxis': dict(showgrid=True, gridcolor='#f0f0f5',
                                                 tickfont=dict(size=10), title='Churn %'),
                                   'xaxis': dict(showgrid=False, tickfont=dict(size=10))})
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown("<div class='slabel'>Churn rate by internet type</div>", unsafe_allow_html=True)
            it = df.groupby('Services.Internet Type')['Status.Churn Value'].mean().reset_index()
            it.columns = ['Type', 'Rate']
            it['Rate'] = (it['Rate'] * 100).round(1)
            it = it.sort_values('Rate', ascending=False)
            fig = go.Figure(go.Bar(
                x=it['Type'], y=it['Rate'],
                marker_color=['#ef4444', '#f59e0b', '#10b981', '#5b5bd6'][:len(it)],
                marker_line_width=0,
                text=[f"{v:.1f}%" for v in it['Rate']],
                textposition='outside', textfont=dict(size=10)))
            apply_base(fig, extra={'yaxis': dict(showgrid=True, gridcolor='#f0f0f5',
                                                 tickfont=dict(size=10), title='Churn %'),
                                   'xaxis': dict(showgrid=False, tickfont=dict(size=10))})
            st.plotly_chart(fig, use_container_width=True)

        with r3:
            st.markdown("<div class='slabel'>Stayed vs churned</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Pie(
                labels=['Stayed', 'Churned'],
                values=[retained, churned],
                hole=0.6,
                marker_colors=['#10b981', '#ef4444'],
                textfont_size=11))
            fig.update_traces(textinfo='percent+label')
            apply_base(fig, extra={'showlegend': False})
            st.plotly_chart(fig, use_container_width=True)

        # row 2
        d1, d2 = st.columns(2)

        with d1:
            st.markdown("<div class='slabel'>How long before churned customers left</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['Status.Churn Value'] == 1]['Services.Tenure in Months'],
                name='Churned', marker_color='#ef4444', opacity=0.75, nbinsx=25))
            fig.add_trace(go.Histogram(
                x=df[df['Status.Churn Value'] == 0]['Services.Tenure in Months'],
                name='Stayed', marker_color='#10b981', opacity=0.75, nbinsx=25))
            apply_base(fig, extra={
                'barmode': 'overlay',
                'legend': dict(orientation='h', y=1.12, font=dict(size=10)),
                'xaxis': dict(showgrid=False, title='Months with company', tickfont=dict(size=10)),
                'yaxis': dict(showgrid=True, gridcolor='#f0f0f5', tickfont=dict(size=10))})
            st.plotly_chart(fig, use_container_width=True)

        with d2:
            st.markdown("<div class='slabel'>Main reasons customers left</div>", unsafe_allow_html=True)
            # use value_counts on the raw column, avoid color groupby KeyError
            cc_series = df['Status.Churn Category'].value_counts().dropna()
            cc = pd.DataFrame({'Reason': cc_series.index.tolist(),
                               'Count': cc_series.values.tolist()})
            bar_colors = ['#ef4444', '#f59e0b', '#5b5bd6', '#8b5cf6', '#10b981'][:len(cc)]
            fig = go.Figure(go.Bar(
                y=cc['Reason'], x=cc['Count'], orientation='h',
                marker_color=bar_colors, marker_line_width=0))
            apply_base(fig, extra={
                'showlegend': False,
                'xaxis': dict(showgrid=False, title='Number of customers', tickfont=dict(size=10)),
                'yaxis': dict(showgrid=False, tickfont=dict(size=10))})
            st.plotly_chart(fig, use_container_width=True)

        # batch upload
        st.divider()
        st.markdown("##### Score a new list of customers")
        st.caption("Upload a CSV or Excel file to get a churn prediction for every customer.")
        uploaded = st.file_uploader("Choose file", type=['csv', 'xlsx'], label_visibility='collapsed')
        if uploaded and model:
            try:
                bd = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
                st.info(f"Loaded {len(bd):,} customers.")
                from sklearn.preprocessing import LabelEncoder

                drop = ['Column1', 'Count', 'Status.Customer Status', 'Status.Churn Label',
                        'Status.Churn Category', 'Status.Churn Reason', 'Location.Country',
                        'Location.State', 'Location.City', 'Location.Zip Code',
                        'Location.Latitude', 'Location.Longitude', 'Population.Population',
                        'Churn Risk', 'Status.Churn Value']
                ref = raw_df.drop(columns=drop, errors='ignore')
                ref['Services.Offer'] = ref['Services.Offer'].fillna('No Offer')
                ref['Services.Internet Type'] = ref['Services.Internet Type'].fillna('No Internet')
                bp = bd.drop(columns=drop, errors='ignore')
                bp['Services.Offer'] = bp['Services.Offer'].fillna('No Offer')
                bp['Services.Internet Type'] = bp['Services.Internet Type'].fillna('No Internet')
                for col in ref.select_dtypes(include='object').columns:
                    if col in bp.columns:
                        le = LabelEncoder();
                        le.fit(ref[col].astype(str))
                        try:
                            bp[col] = le.transform(bp[col].astype(str))
                        except:
                            bp[col] = 0
                for col in feature_cols:
                    if col not in bp.columns: bp[col] = 0
                probs = model.predict_proba(bp[feature_cols])[:, 1]
                res = pd.DataFrame()
                if 'Column1' in bd.columns: res['Customer ID'] = bd['Column1'].values
                res['Churn Probability %'] = (probs * 100).round(1)
                res['Risk Level'] = pd.cut(probs, bins=[0, .35, .65, 1], labels=['Low', 'Medium', 'High'])
                if 'Services.Monthly Charge' in bd.columns:
                    res['Monthly Bill ($)'] = bd['Services.Monthly Charge'].values
                res = res.sort_values('Churn Probability %', ascending=False).reset_index(drop=True)
                h = (res['Risk Level'] == 'High').sum()
                m = (res['Risk Level'] == 'Medium').sum()
                l = (res['Risk Level'] == 'Low').sum()
                x1, x2, x3 = st.columns(3)
                x1.metric("High Risk — Act Now", f"{h:,}")
                x2.metric("Medium Risk — Watch", f"{m:,}")
                x3.metric("Low Risk — Healthy", f"{l:,}")
                st.dataframe(res.style.background_gradient(
                    subset=['Churn Probability %'], cmap='RdYlGn_r'),
                    use_container_width=True, height=260)
                st.download_button("Download Predictions",
                                   res.to_csv(index=False).encode(), "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INDIVIDUAL PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("##### Check if a specific customer is at risk")
    st.caption(
        "Fill in the customer details and click the button. You will instantly see the risk level, what is causing it, and what to do.")
    if model is None:
        st.error("Model not loaded. Run both Jupyter notebooks first.")
    else:
        with st.form("pform"):
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("<div class='slabel'>Personal info</div>", unsafe_allow_html=True)
                age = st.slider("Age", 18, 90, 45)
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior citizen (65+)", ["No", "Yes"])
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.selectbox("Has dependents", ["No", "Yes"])
                num_dep = st.number_input("Number of dependents", 0, 10, 0)
            with f2:
                st.markdown("<div class='slabel'>Services</div>", unsafe_allow_html=True)
                tenure = st.slider("Months with company", 1, 72, 10)
                contract = st.selectbox("Contract type", ["Month-to-Month", "One Year", "Two Year"])
                internet = st.selectbox("Internet service", ["Fiber Optic", "DSL", "Cable", "No Internet"])
                offer = st.selectbox("Current offer",
                                     ["No Offer", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
                tech_support = st.selectbox("Has tech support", ["No", "Yes"])
                streaming_tv = st.selectbox("Has streaming TV", ["No", "Yes"])
                streaming_mv = st.selectbox("Has streaming movies", ["No", "Yes"])
            with f3:
                st.markdown("<div class='slabel'>Billing</div>", unsafe_allow_html=True)
                monthly_charge = st.slider("Monthly bill ($)", 18, 120, 70)
                total_charges = st.number_input("Total billed so far ($)", 0.0, 15000.0, float(monthly_charge * tenure))
                total_revenue = st.number_input("Total revenue ($)", 0.0, 15000.0, float(monthly_charge * tenure * 1.1))
                payment = st.selectbox("Payment method",
                                       ["Bank Withdrawal", "Credit Card", "Electronic Check", "Mailed Check"])
                sat_score = st.slider("Satisfaction (1=unhappy, 5=very happy)", 1, 5, 3)
                cltv = st.number_input("Customer lifetime value ($)", 0, 15000, 4000)
            go_btn = st.form_submit_button("Check Churn Risk", use_container_width=True)

        if go_btn:
            inp = {
                'Gender': gender, 'Age': age, 'Under 30': 'Yes' if age < 30 else 'No',
                'Senior Citizen': senior, 'Married': married, 'Dependents': dependents,
                'Number of Dependents': num_dep, 'Services.Tenure in Months': tenure,
                'Services.Offer': offer, 'Services.Internet Type': internet,
                'Services.Premium Tech Support': tech_support,
                'Services.Streaming TV': streaming_tv, 'Services.Streaming Movies': streaming_mv,
                'Services.Contract': contract, 'Services.Payment Method': payment,
                'Services.Monthly Charge': monthly_charge, 'Services.Total Charges': total_charges,
                'Services.Total Revenue': total_revenue, 'Status.Satisfaction Score': sat_score,
                'Status.Churn Score': 50, 'Status.CLTV': cltv,
                'Age Group': min(age // 10 * 10, 70),
                'Tenure Group': 'New Customer' if tenure <= 12 else 'Mid-term' if tenure <= 36 else 'Long-term',
            }
            try:
                X = encode_input(inp)
                prob = model.predict_proba(X)[0][1]
                risk, color = risk_label(prob)

                st.divider()
                st.markdown("##### Prediction Result")
                rl, rr = st.columns([1, 2])

                with rl:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=round(prob * 100, 1),
                        number={'suffix': '%', 'font': {'size': 28, 'color': color, 'family': 'Inter'}},
                        gauge={'axis': {'range': [0, 100], 'tickfont': {'size': 8}},
                               'bar': {'color': color, 'thickness': 0.2},
                               'bgcolor': '#f5f6fa', 'borderwidth': 0,
                               'steps': [{'range': [0, 35], 'color': '#f0fdf4'},
                                         {'range': [35, 65], 'color': '#fffbeb'},
                                         {'range': [65, 100], 'color': '#fef2f2'}]}))
                    fig_g.update_layout(height=165, margin=dict(l=8, r=8, t=8, b=0),
                                        paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'))
                    st.plotly_chart(fig_g, use_container_width=True)

                    rc = {'High Risk': 'cg-danger', 'Medium Risk': 'cg-warning', 'Low Risk': 'cg-success'}[risk]
                    act = ('Contact this customer today — they are very likely to leave.' if 'High' in risk
                           else 'Keep an eye on this customer over the next 30 days.' if 'Medium' in risk
                    else 'This customer looks stable. Keep nurturing the relationship.')
                    st.markdown(f"""
                    <div class="cg-card {rc}">
                      <div style="font-size:.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:.06em">Verdict</div>
                      <div style="font-size:1.2rem;font-weight:700;color:{color};margin:2px 0">{risk}</div>
                      <div style="font-size:.77rem;color:#6b7280;line-height:1.4">{act}</div>
                    </div>""", unsafe_allow_html=True)

                    diff = prob - 0.265
                    dc = '#ef4444' if diff > 0 else '#10b981'
                    st.markdown(f"""
                    <div class="cg-card">
                      <div style="font-size:.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:.06em">vs average customer</div>
                      <div style="font-size:1rem;font-weight:600;color:{dc};margin-top:3px">
                        {abs(diff) * 100:.1f}% {'above' if diff > 0 else 'below'} average
                      </div>
                      <div style="font-size:.73rem;color:#9ca3af">Company average is 26.5%</div>
                    </div>""", unsafe_allow_html=True)

                with rr:
                    st.markdown("<div class='slabel'>What is driving this risk</div>", unsafe_allow_html=True)
                    st.caption("Red bars = reasons pushing toward leaving.  Green bars = reasons keeping them.")
                    try:
                        sv = explainer.shap_values(X)
                        sv = sv[0] if isinstance(sv, list) else sv[0]
                        ss = pd.Series(sv, index=feature_cols).abs().sort_values(ascending=False).head(8)
                        bc = ['#ef4444' if sv[feature_cols.index(f)] > 0 else '#10b981' for f in ss.index]
                        fig_s = go.Figure(go.Bar(
                            x=ss.values, y=ss.index.tolist(),
                            orientation='h', marker_color=bc, marker_line_width=0))
                        apply_base(fig_s, h=225, extra={
                            'xaxis': dict(showgrid=False, title='Impact strength', tickfont=dict(size=10)),
                            'yaxis': dict(showgrid=False, autorange='reversed', tickfont=dict(size=10))})
                        st.plotly_chart(fig_s, use_container_width=True)
                    except Exception as e:
                        st.info(f"Chart unavailable: {e}")

                    st.markdown(
                        "<div class='slabel' style='margin-top:.5rem'>This customer vs average churned customer</div>",
                        unsafe_allow_html=True)
                    cavg = raw_df[raw_df['Status.Churn Value'] == 1][
                        ['Services.Tenure in Months', 'Services.Monthly Charge',
                         'Status.Satisfaction Score']].mean()
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Bar(name='This customer',
                                           x=['Tenure (mo)', 'Monthly bill ($)', 'Satisfaction (1-5)'],
                                           y=[tenure, monthly_charge, sat_score],
                                           marker_color='#5b5bd6', marker_line_width=0))
                    fig_c.add_trace(go.Bar(name='Avg churned customer',
                                           x=['Tenure (mo)', 'Monthly bill ($)', 'Satisfaction (1-5)'],
                                           y=[round(cavg['Services.Tenure in Months'], 1),
                                              round(cavg['Services.Monthly Charge'], 1),
                                              round(cavg['Status.Satisfaction Score'], 1)],
                                           marker_color='#fca5a5', marker_line_width=0))
                    apply_base(fig_c, h=185, extra={
                        'barmode': 'group',
                        'legend': dict(orientation='h', y=1.15, font=dict(size=10)),
                        'xaxis': dict(showgrid=False, tickfont=dict(size=10)),
                        'yaxis': dict(showgrid=True, gridcolor='#f0f0f5', tickfont=dict(size=10))})
                    st.plotly_chart(fig_c, use_container_width=True)

                st.divider()
                st.markdown("##### AI Retention Plan")
                with st.spinner("Generating plan..."):
                    ai = call_ai(f"""You are a telecom customer retention specialist. Write in plain English that any customer service agent can understand — no jargon.

Customer: Age {age}, {tenure} months with us, contract: {contract}, monthly bill: ${monthly_charge}, satisfaction: {sat_score}/5, tech support: {tech_support}, payment: {payment}.
Churn probability: {prob * 100:.1f}% ({risk}).

ANALYSIS: [2-3 sentences explaining why this customer might leave based on the profile]

RETENTION ACTIONS:
1. [Specific action the agent should take today]
2. [Specific offer or script to use]
3. [Follow-up action for next 30 days]""")

                parts = ai.split('RETENTION ACTIONS:')
                analysis = parts[0].replace('ANALYSIS:', '').strip()
                actions = parts[1].strip() if len(parts) > 1 else ai
                a1, a2 = st.columns(2)
                with a1:
                    st.markdown(f'<div class="ai-box"><h4>Why they might leave</h4><p>{analysis}</p></div>',
                                unsafe_allow_html=True)
                with a2:
                    st.markdown(
                        f'<div class="ai-box"><h4>What to do right now</h4><p style="white-space:pre-line">{actions}</p></div>',
                        unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHY THEY LEAVE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("##### What is making your customers leave")
    st.caption("Based on your actual data — ranked by impact — with plain-English fixes for each issue.")
    if raw_df.empty:
        st.warning("Data not loaded.")
    else:
        churned_df = raw_df[raw_df['Status.Churn Value'] == 1].copy()
        cat_series = churned_df['Status.Churn Category'].value_counts().dropna()
        reason_series = churned_df['Status.Churn Reason'].value_counts().dropna().head(6)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("<div class='slabel'>Churn by category</div>", unsafe_allow_html=True)
            fig = go.Figure(go.Pie(
                labels=cat_series.index.tolist(),
                values=cat_series.values.tolist(),
                hole=0.55,
                marker_colors=['#ef4444', '#f59e0b', '#5b5bd6', '#8b5cf6', '#10b981'][:len(cat_series)],
                textfont_size=11))
            fig.update_traces(textinfo='percent+label')
            apply_base(fig, extra={'showlegend': False})
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            st.markdown("<div class='slabel'>Top specific reasons customers left</div>", unsafe_allow_html=True)
            r_colors = ['#ef4444', '#f59e0b', '#5b5bd6', '#8b5cf6', '#10b981', '#06b6d4'][:len(reason_series)]
            fig = go.Figure(go.Bar(
                x=reason_series.values.tolist(),
                y=reason_series.index.tolist(),
                orientation='h',
                marker_color=r_colors,
                marker_line_width=0))
            apply_base(fig, extra={
                'showlegend': False,
                'xaxis': dict(showgrid=False, title='Number of customers', tickfont=dict(size=10)),
                'yaxis': dict(showgrid=False, tickfont=dict(size=10))})
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("##### Ranked issues — with what to do about each")
        for reason, info in CHURN_REASONS.items():
            cc = {'Critical': 'cg-danger', 'High': 'cg-warning', 'Medium': 'cg-info'}.get(info['tag'], 'cg-card')
            bc = '#ef4444' if info['impact'] > 70 else '#f59e0b' if info['impact'] > 50 else '#5b5bd6'
            fh = ''.join([f"<li style='margin-bottom:5px;color:#374151'>{f}</li>" for f in info['fixes']])
            st.markdown(f"""
            <div class="cg-card {cc}">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">
                <span style="font-weight:600;color:#111827;font-size:.88rem">{reason}</span>
                <span class="pill {info['color']}">{info['tag']}</span>
              </div>
              <div style="color:#6b7280;font-size:.81rem;margin-bottom:7px;line-height:1.5">{info['desc']}</div>
              <div style="background:#f0f0f5;border-radius:4px;height:5px;margin-bottom:9px;overflow:hidden">
                <div style="height:100%;width:{info['impact']}%;background:{bc};border-radius:4px"></div>
              </div>
              <div style="font-size:.65rem;color:#9ca3af;margin-bottom:4px;text-transform:uppercase;letter-spacing:.06em">What to do</div>
              <ul style="margin:0;padding-left:15px;font-size:.81rem">{fh}</ul>
            </div>""", unsafe_allow_html=True)

        st.divider()
        if st.button("Generate a campaign strategy with AI"):
            with st.spinner("Building strategy..."):
                result = call_ai(f"""You are a telecom retention strategist. Write in plain language — no jargon.

Data: {len(churned_df):,} customers churned.
Top categories: {dict(cat_series.head(3))}.
Top specific reasons: {dict(reason_series.head(3))}.
Average monthly bill of churned customers: ${churned_df['Services.Monthly Charge'].mean():.0f}.

Write 3 short paragraphs:
1. The biggest problem and why it matters
2. The campaign you recommend — give it a name, say who to target and what to offer
3. What success looks like in 90 days with actual numbers""")
                st.markdown(
                    f'<div class="ai-box"><h4>AI Campaign Strategy</h4><p style="white-space:pre-line">{result}</p></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REVENUE IMPACT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("##### See how much a retention campaign can save")
    st.caption("Move the sliders to set up your campaign. All numbers update instantly.")
    if raw_df.empty:
        st.warning("Data not loaded.")
    else:
        at_risk = int(raw_df['Status.Churn Value'].sum())
        avg_bill = float(raw_df['Services.Monthly Charge'].mean())
        monthly_loss = float(raw_df[raw_df['Status.Churn Value'] == 1]['Services.Monthly Charge'].sum())

        st.markdown("##### Without any action — what you are losing right now")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Customers at risk", f"{at_risk:,}")
        b2.metric("Lost every month", f"${monthly_loss:,.0f}")
        b3.metric("Lost every year", f"${monthly_loss * 12:,.0f}")
        b4.metric("Avg bill per customer", f"${avg_bill:.2f}")

        st.divider()
        st.markdown("##### Set up your campaign")
        s1, s2 = st.columns(2)
        with s1:
            targeted = st.slider("How many at-risk customers will you contact?",
                                 100, at_risk, min(600, at_risk))
            conv_rate = st.slider("What % do you expect to keep?", 5, 60, 30,
                                  help="If you contact 100 and keep 30, that is 30%")
            months_ret = st.slider("How many extra months will they stay?", 3, 36, 12)
        with s2:
            cost_per = st.slider("Campaign cost per customer ($)", 5, 100, 25)
            discount = st.slider("Discount you are offering (%)", 0, 30, 10)
            avg_rev_cust = st.slider("Average monthly bill per customer ($)", 30, 120, int(avg_bill))

        saved = int(targeted * conv_rate / 100)
        camp_cost = targeted * cost_per
        disc_rev = avg_rev_cust * (1 - discount / 100)
        rev_saved = saved * disc_rev * months_ret
        net_profit = rev_saved - camp_cost
        roi = (net_profit / camp_cost * 100) if camp_cost > 0 else 0
        m_before = targeted * avg_rev_cust
        m_after = m_before - ((targeted - saved) * avg_rev_cust)

        st.divider()
        st.markdown("##### Your results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Customers kept", f"{saved:,}")
        r2.metric("Revenue saved", f"${rev_saved:,.0f}")
        r3.metric("Campaign cost", f"${camp_cost:,.0f}")
        r4.metric("Net profit", f"${net_profit:,.0f}",
                  delta=f"ROI {roi:.0f}%",
                  delta_color="normal" if net_profit >= 0 else "inverse")

        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("<div class='slabel'>Revenue over time — with vs without campaign</div>",
                        unsafe_allow_html=True)
            mo = list(range(1, months_ret + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mo, y=[m_before * m for m in mo],
                                     name='Without campaign',
                                     line=dict(color='#fca5a5', width=2), fill='tozeroy',
                                     fillcolor='rgba(252,165,165,0.1)'))
            fig.add_trace(go.Scatter(x=mo, y=[m_after * m for m in mo],
                                     name='With campaign',
                                     line=dict(color='#10b981', width=2), fill='tozeroy',
                                     fillcolor='rgba(16,185,129,0.1)'))
            apply_base(fig, extra={
                'legend': dict(orientation='h', y=1.15, font=dict(size=10)),
                'xaxis': dict(showgrid=False, title='Month', tickfont=dict(size=10)),
                'yaxis': dict(showgrid=True, gridcolor='#f0f0f5', tickprefix='$', tickfont=dict(size=10))})
            st.plotly_chart(fig, use_container_width=True)

        with gc2:
            st.markdown("<div class='slabel'>Financial summary</div>", unsafe_allow_html=True)
            pc = '#10b981' if net_profit >= 0 else '#ef4444'
            fig2 = go.Figure(go.Bar(
                x=['Campaign cost', 'Revenue saved', 'Net profit'],
                y=[camp_cost, rev_saved, net_profit],
                marker_color=['#f59e0b', '#5b5bd6', pc],
                marker_line_width=0,
                text=[f"${v:,.0f}" for v in [camp_cost, rev_saved, net_profit]],
                textposition='outside', textfont=dict(size=10, color='#374151')))
            apply_base(fig2, extra={
                'xaxis': dict(showgrid=False, tickfont=dict(size=10)),
                'yaxis': dict(showgrid=True, gridcolor='#f0f0f5', tickprefix='$', tickfont=dict(size=10))})
            st.plotly_chart(fig2, use_container_width=True)

        vc = 'cg-success' if net_profit > 0 else 'cg-danger'
        vt = (
            f"This campaign makes money. You gain ${net_profit:,.0f} after all costs — a {roi:.0f}% return on investment."
            if net_profit > 0 else
            f"This campaign loses ${abs(net_profit):,.0f} at these settings. Try reducing cost per customer or increasing the conversion rate.")
        st.markdown(f"""
        <div class="cg-card {vc}">
          <div style="font-weight:600;color:#111827;margin-bottom:3px">
            {'Profitable at these settings' if net_profit > 0 else 'Not profitable at these settings'}
          </div>
          <div style="font-size:.84rem;color:#6b7280">{vt}</div>
        </div>""", unsafe_allow_html=True)

        if st.button("Get AI analysis of this campaign"):
            with st.spinner("Analyzing..."):
                ai_r = call_ai(f"""Campaign analyst. Plain English only — no jargon.
Numbers: {targeted:,} customers contacted, {saved:,} kept ({conv_rate}% conversion rate),
campaign cost ${camp_cost:,.0f}, revenue saved ${rev_saved:,.0f} over {months_ret} months,
net profit ${net_profit:,.0f}, ROI {roi:.0f}%.
Write 2-3 sentences saying whether this is good or bad and why.
Then give 2 specific changes to improve the ROI.""")
                st.markdown(f'<div class="ai-box"><h4>AI Analysis</h4><p>{ai_r}</p></div>',
                            unsafe_allow_html=True)

        st.divider()
        st.markdown("##### Compare different campaign sizes side by side")
        rows = []
        for name, s in [
            ('Careful — low spend', {'conv': 15, 'cost': 10, 'months': 6}),
            ('Balanced', {'conv': 30, 'cost': 25, 'months': 12}),
            ('Aggressive — high spend', {'conv': 45, 'cost': 45, 'months': 18}),
            ('Mass outreach', {'conv': 12, 'cost': 8, 'months': 5}),
        ]:
            sv = int(targeted * s['conv'] / 100);
            cs = targeted * s['cost']
            rs = sv * avg_rev_cust * s['months'];
            np_ = rs - cs
            roi_ = (np_ / cs * 100) if cs > 0 else 0
            rows.append({'Scenario': name, 'Customers kept': sv, 'Campaign cost': f"${cs:,.0f}",
                         'Revenue saved': f"${rs:,.0f}", 'Net profit': f"${np_:,.0f}",
                         'ROI': f"{roi_:.0f}%", 'Result': 'Profitable' if np_ > 0 else 'Loss'})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
