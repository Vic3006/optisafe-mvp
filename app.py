import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =====================================================================
# PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(page_title="OptiSafe MVP Dashboard", page_icon="🛡️", layout="wide")

# OptiSafe Brand Colors
colors = {"navy": "#1A2A3A", "orange": "#F26D21", "gray": "#B0B0B0", 
          "green": "#28a745", "yellow": "#ffc107", "red": "#dc3545"}
sns.set_theme(style="whitegrid")

# =====================================================================
# DATA LOADING
# =====================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('wearable_data_full_day_bueno.csv')
        df.columns = df.columns.str.strip()
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['hour_numeric'] = df['timestamp_dt'].dt.hour + df['timestamp_dt'].dt.minute/60.0
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'wearable_data_full_day_bueno.csv' is in the repository.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Extract only the active work shift
df_work = df[df['pieces_assembled_hour'] > 0].copy()

# =====================================================================
# SIDEBAR CONTROLS (INTERACTIVITY)
# =====================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2059/2059738.png", width=80) # Generic shield icon
st.sidebar.title("OptiSafe Controls")
st.sidebar.markdown("---")

# Dropdown to select an employee
employee_list = df_work['user'].unique()
selected_user = st.sidebar.selectbox("Select Employee to Monitor:", employee_list)

st.sidebar.markdown("---")
st.sidebar.info("This is the interactive OptiSafe MVP. Select different employees to see how the AI adapts to their unique biometric baselines.")

# Filter data for the selected user
df_user = df_work[df_work['user'] == selected_user].copy()

# =====================================================================
# TOP METRICS DASHBOARD
# =====================================================================
st.title("🛡️ OptiSafe: Real-Time Supervisor Dashboard")
st.markdown("Monitor team fatigue, predict accidents, and optimize productivity.")

# Calculate dynamic metrics for the selected user
current_stress = df_user['stress_level'].iloc[-1]
avg_hrv = df_user['hrv'].mean()
total_pieces = df_user['pieces_assembled_hour'].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Workers", len(employee_list))
col2.metric(f"Current Stress ({selected_user})", f"{current_stress:.0f}/100", delta="High Risk" if current_stress > 70 else "Normal", delta_color="inverse")
col3.metric(f"Avg HRV ({selected_user})", f"{avg_hrv:.1f} ms")
col4.metric(f"Total Pieces Assembled", total_pieces)

st.markdown("---")

# =====================================================================
# MODULE 1 & 4: INDIVIDUAL MONITORING & ROI (SIDE BY SIDE)
# =====================================================================
colA, colB = st.columns(2)

with colA:
    st.subheader(f"Biometric Alert System: {selected_user}")
    
    df_user['stress_trend'] = df_user['stress_level'].rolling(window=2, min_periods=1).mean()
    stress_mean = df_user['stress_level'].mean()
    stress_std = df_user['stress_level'].std()
    threshold = stress_mean + stress_std
    df_user['alert'] = df_user['stress_level'] > threshold

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df_user['hour_numeric'], df_user['stress_level'], color=colors["gray"], alpha=0.5, label='Raw Stress')
    ax1.plot(df_user['hour_numeric'], df_user['stress_trend'], color=colors["navy"], linewidth=3, label='Accumulated Trend')
    ax1.axhline(threshold, color=colors["orange"], linestyle='--', linewidth=2, label='Critical Intervention Threshold')
    ax1.fill_between(df_user['hour_numeric'], threshold, df_user['stress_level'].max() + 5, color=colors["orange"], alpha=0.1)
    
    alerts = df_user[df_user['alert']]
    ax1.scatter(alerts['hour_numeric'], alerts['stress_trend'], color=colors["red"], s=100, zorder=5, label='Break Recommended')
    
    ax1.set_xlabel('Hour of the Day (24h)', fontsize=10)
    ax1.set_ylabel('Stress Level', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    st.pyplot(fig1)

with colB:
    st.subheader(f"Productivity ROI Simulation: {selected_user}")
    
    actual_pieces = df_user['pieces_assembled_hour'].values
    shift_hours = np.arange(1, len(actual_pieces) + 1)
    actual_stress = df_user['stress_level'].values

    optisafe_pieces = []
    alert_triggered = False
    recovered_pieces = 0

    for i in range(len(actual_pieces)):
        current_piece = actual_pieces[i]
        stress = actual_stress[i]
        
        if alert_triggered:
            rescue = int(np.random.uniform(2, 5))
            recovered_pieces += rescue
            optisafe_pieces.append(current_piece + rescue)
        else:
            optisafe_pieces.append(current_piece)
            
        if stress > threshold:
            alert_triggered = True

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(shift_hours, optisafe_pieces, marker='o', color=colors["navy"], linewidth=2, label='With OptiSafe (Proactive)')
    ax4.plot(shift_hours, actual_pieces, marker='X', color=colors["orange"], linestyle='--', linewidth=2, label='Status Quo (Fatigue Drop)')
    ax4.fill_between(shift_hours, optisafe_pieces, actual_pieces, color=colors["green"], alpha=0.15, label=f'ROI: +{recovered_pieces} Pieces')
    
    ax4.set_xlabel('Elapsed Shift Hour', fontsize=10)
    ax4.set_ylabel('Pieces Assembled / Hour', fontsize=10)
    ax4.legend(loc='lower left', fontsize=8)
    st.pyplot(fig4)

st.markdown("---")

# =====================================================================
# MODULE 2 & 3: TEAM DASHBOARD & SCIENTIFIC VALIDATION
# =====================================================================
colC, colD = st.columns(2)

with colC:
    st.subheader("Team Risk Dashboard (Fatigue Risk Score)")
    
    profiles = df_work.groupby('user').agg(
        avg_stress=('stress_level', 'mean'),
        avg_hrv=('hrv', 'mean'),
        avg_sleep=('sleep_quality', 'mean')
    ).reset_index()

    profiles['risk_score'] = (profiles['avg_stress'] * 0.45) + ((100 - profiles['avg_hrv']) * 0.35) + ((100 - profiles['avg_sleep']) * 0.20)
    profiles = profiles.sort_values('risk_score').reset_index(drop=True)

    num_users = len(profiles)
    bar_colors = [colors["green"] if i < num_users/3 else colors["yellow"] if i < (num_users/3)*2 else colors["red"] for i in range(num_users)]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars = ax2.bar(profiles['user'], profiles['risk_score'], color=bar_colors)
    
    for bar, score, hrv in zip(bars, profiles['risk_score'], profiles['avg_hrv']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'FRS: {score:.1f}\nHRV: {hrv:.1f}', 
                 ha='center', va='bottom', color='black', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel('Risk Index (0-100)', fontsize=10)
    ax2.set_ylim(0, max(profiles['risk_score']) + 20)
    st.pyplot(fig2)

with colD:
    st.subheader("Scientific Correlation: Sleep vs HRV")
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['sleep_quality'], df['hrv'])

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.regplot(x=df['sleep_quality'], y=df['hrv'], color=colors["navy"],
                scatter_kws={'alpha':0.5, 's': 60, 'color': colors["orange"]},
                line_kws={'label': f'Scientific Correlation ($R^2$={r_value**2:.2f})'}, ax=ax3)
    
    ax3.set_xlabel('Reported Sleep Quality (%)', fontsize=10)
    ax3.set_ylabel('HRV (ms)', fontsize=10)
    ax3.legend(fontsize=8)
    st.pyplot(fig3)

st.markdown("<p style='text-align: center; color: gray;'>OptiSafe MVP v4.0 - Proprietary Dashboard</p>", unsafe_allow_html=True)
