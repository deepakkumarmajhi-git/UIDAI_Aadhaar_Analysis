import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="UIDAI Aadhaar Analytics",
    layout="wide"
)

st.title("📊 UIDAI Aadhaar Enrollment & Update Analytics")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    enrol = pd.read_csv(r"api_data_aadhar_enrolment.csv")
    demo = pd.read_csv(r"api_data_aadhar_demograp.csv")
    bio = pd.read_csv(r"api_data_aadhar_biometric.csv")
    return enrol, demo, bio

enrol, demo, bio = load_data()

# -------------------------
# DATE FIX
# -------------------------
for df in [enrol, demo, bio]:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['month'] = df['date'].dt.to_period('M')

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("🔍 Filters")

state = st.sidebar.selectbox(
    "Select State",
    ["All"] + sorted(enrol['state'].dropna().unique().tolist())
)

if state != "All":
    enrol = enrol[enrol['state'] == state]
    demo = demo[demo['state'] == state]
    bio = bio[bio['state'] == state]

# -------------------------
# EXECUTIVE OVERVIEW
# -------------------------
st.header("📌 Executive Overview")

total_enrol = enrol[['age_0_5','age_5_17','age_18_greater']].sum().sum()
total_demo = demo[['demo_age_5_17','demo_age_17_']].sum().sum()
total_bio = bio[['bio_age_5_17','bio_age_17_']].sum().sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Enrolments", f"{int(total_enrol):,}")
col2.metric("Demographic Updates", f"{int(total_demo):,}")
col3.metric("Biometric Updates", f"{int(total_bio):,}")

st.info(
    "Demographic and biometric updates constitute a significant share of Aadhaar system activity, "
    "indicating sustained post-enrolment demand and the need for continuous infrastructure readiness."
)

# -------------------------
# TEMPORAL PATTERNS & SEASONALITY
# -------------------------
st.header("📈 Temporal Patterns & Seasonality")

monthly_demo = demo.groupby('month')[['demo_age_5_17','demo_age_17_']].sum().sum(axis=1)
monthly_bio = bio.groupby('month')[['bio_age_5_17','bio_age_17_']].sum().sum(axis=1)
compare_df = pd.concat([monthly_demo, monthly_bio], axis=1).reset_index()
compare_df.columns = ['month', 'Demographic Updates', 'Biometric Updates']
compare_df['month'] = compare_df['month'].astype(str)

fig_trend = px.line(
    compare_df,
    x='month',
    y=['Demographic Updates', 'Biometric Updates'],
    markers=True,
    title="Monthly Update Trends"
)
st.plotly_chart(fig_trend, use_container_width=True)

st.write("""
**Observed Pattern:**
- Update activity shows monthly variability, suggesting seasonal or policy-driven effects.
- Demographic updates typically precede biometric updates, indicating phased compliance behavior.
""")

# -------------------------
# ANOMALY DETECTION
# -------------------------
st.header("⚠️ Anomaly Detection")

# Combine demo + bio updates
monthly_total = monthly_demo.add(monthly_bio, fill_value=0)

# Convert index (Period) to string
monthly_total.index = monthly_total.index.astype(str)

# Give the Series a name
monthly_total.name = "Total Updates"

# Reset index to DataFrame
df_monthly_total = monthly_total.reset_index()
df_monthly_total.columns = ['Month', 'Total Updates']

# Z-score for anomaly detection
z_scores = (df_monthly_total['Total Updates'] - df_monthly_total['Total Updates'].mean()) / df_monthly_total['Total Updates'].std()
anomalies = df_monthly_total[z_scores.abs() > 2]

# Plot
fig_anomaly = px.line(
    df_monthly_total,
    x='Month',
    y='Total Updates',
    title="Monthly Aadhaar Update Volume with Anomalies"
)

# Highlight anomalies
fig_anomaly.add_scatter(
    x=anomalies['Month'],
    y=anomalies['Total Updates'],
    mode='markers',
    name='Anomaly',
    marker=dict(size=10, color='red')
)

st.plotly_chart(fig_anomaly, use_container_width=True)


# -------------------------
# PREDICTIVE INDICATORS
# -------------------------
st.header("📊 Short-Term Predictive Indicators")

rolling_avg = monthly_total.rolling(3).mean()
trend_df = pd.concat([monthly_total, rolling_avg], axis=1).reset_index()
trend_df.columns = ['Month', 'Actual', '3-Month Rolling Avg']

fig_pred = px.line(
    trend_df,
    x='Month',
    y=['Actual', '3-Month Rolling Avg'],
    title="Short-Term Predictive Indicator (Rolling Trend)"
)
st.plotly_chart(fig_pred, use_container_width=True)

st.success(
    "Rolling averages act as early indicators of sustained demand increases, "
    "enabling proactive capacity planning before peak loads occur."
)

# -------------------------
# AGE-WISE ENROLLMENT DISTRIBUTION
# -------------------------
st.header("👶 Age-wise Enrollment Distribution")

age_dist = enrol[['age_0_5','age_5_17','age_18_greater']].sum().reset_index()
age_dist.columns = ['Age Group', 'Count']

fig_age = px.pie(
    age_dist,
    names='Age Group',
    values='Count',
    hole=0.4,
    title="Age-wise Enrollment Distribution"
)
st.plotly_chart(fig_age, use_container_width=True)

# -------------------------
# SOLUTION FRAMEWORK
# -------------------------
st.header("🛠️ Decision Support Framework")

solution_df = pd.DataFrame({
    "Observation": [
        "High update stress in specific states",
        "Seasonal demand spikes",
        "Anomalous surges in update volume"
    ],
    "Risk": [
        "Service delays & congestion",
        "Understaffing during peak months",
        "System overload or misuse"
    ],
    "Recommended Action": [
        "Deploy mobile enrollment units",
        "Temporary staffing & extended hours",
        "Real-time monitoring & alerts"
    ]
})

st.dataframe(solution_df)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("**UIDAI Data Hackathon Dashboard** | Built with Streamlit & Plotly")
st.markdown("**PRANGYA SREE PATTANAYAK** |**DEEPAK KUMAR MAJHI** | **ISHU ANAND** | **MIHIR KUMAR PANIGRAHI** |")
