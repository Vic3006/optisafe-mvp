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
# AUTHENTICATION SETUP (THE "BOUNCER")
# =====================================================================
# Hardcoded credentials for the MVP
USER_CREDENTIALS = {
    "admin": "OptiSafe2026",
    "supervisor": "demo123"
}

# Initialize the session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login():
    st.markdown("<h1 style='text-align: center;'>🛡️ OptiSafe Portal Login</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Enter your supervisor credentials to access the dashboard.</p>", unsafe_allow_html=True)
    
    # Create columns to center the login box
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("---")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password") # Hides the text as dots
        
        if st.button("Log In", use_container_width=True):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state['logged_in'] = True
                st.rerun() # Refreshes the app to clear the login screen
            else:
                st.error("🚨 Invalid username or password. Please try again.")
        st.markdown("---")

def logout():
    st.session_state['logged_in'] = False
    st.rerun()

# =====================================================================
# MAIN DASHBOARD FUNCTION
# =====================================================================
def run_dashboard():
    # DATA LOADING
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv('wearable_data_100_employees.csv')
            df.columns = df.columns.str.strip()
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='mixed')
            df['hour_numeric'] = df['timestamp_dt'].dt.hour + df['timestamp_dt'].dt.minute/60.0
            return df
        except FileNotFoundError:
            st.error("Data file not found. Please ensure 'wearable_data_100_employees.csv' is in your GitHub repository.")
            return pd.DataFrame()

    df = load_data()

    if df.empty:
        st.stop()

    df_work = df[df['pieces_assembled_hour'] > 0].copy()

    # SIDEBAR CONTROLS
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2059/2059738.png", width=80) 
    st.sidebar.title("OptiSafe Controls")
    
    # Add Logout Button at the top of the sidebar
    if st.sidebar.button("🚪 Log Out", type="primary"):
        logout()
        
    st.sidebar.markdown("---")

    employee_list = df_work['user'].unique()
    selected_user = st.sidebar.selectbox("Select Employee to Monitor:", employee_list)

    st.sidebar.markdown("---")
    st.sidebar.info("This is the interactive OptiSafe MVP. Select different employees to see how the AI adapts to their unique biometric baselines.")

    df_user = df_work[df_work['user'] == selected_user].copy()

    # TOP METRICS DASHBOARD
    st.title("🛡️ OptiSafe: Real-Time Supervisor Dashboard")
    st.markdown("Monitor team fatigue, predict accidents, and optimize productivity.")

    current_stress = df_user['stress_level'].iloc[-1]
    avg_hrv = df_user['hrv'].mean()
    total_pieces = df_user['pieces_assembled_hour'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Workers Monitored", len(employee_list))
    col2.metric(f"Current Stress ({selected_user})", f"{current_stress:.0f}/100", delta="High Risk" if current_stress > 70 else "Normal", delta_color="inverse")
    col3.metric(f"Avg HRV ({selected_user})", f"{avg_hrv:.1f} ms")
    col4.metric(f"Total Pieces Assembled", int(total_pieces))

    st.markdown("---")

    # MODULE 1 & 4
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
        
        max_stress_plot = max(100, df_user['stress_level'].max() + 5)
        ax1.fill_between(df_user['hour_numeric'], threshold, max_stress_plot, color=colors["orange"], alpha=0.1)
        
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
                rescue = int(np.random.uniform(-5, -3))
                recovered_pieces += rescue
                optisafe_pieces.append(current_piece + rescue)
            else:
                optisafe_pieces.append(current_piece)
                
            if stress > threshold:
                alert_triggered = True

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.plot(shift_hours, optisafe_pieces, marker='o', color=colors["navy"], linewidth=2, label='With OptiSafe (Proactive)')
        ax4.plot(shift_hours, actual_pieces, marker='X', color=colors["navy"], linestyle='--', linewidth=2, label='Status Quo (Fatigue Drop)')
        ax4.fill_between(shift_hours, optisafe_pieces, actual_pieces, color=colors["green"], alpha=0.15, label=f'ROI: +{recovered_pieces} Pieces')
        
        ax4.set_xlabel('Elapsed Shift Hour', fontsize=10)
        ax4.set_ylabel('Pieces Assembled / Hour', fontsize=10)
        ax4.legend(loc='lower left', fontsize=8)
        st.pyplot(fig4)

    st.markdown("---")

    # MODULE 2 & 3
    colC, colD = st.columns(2)

    with colC:
        st.subheader("Team Risk Dashboard (Top 20 Highest Risk)")
        
        profiles = df_work.groupby('user').agg(
            avg_stress=('stress_level', 'mean'),
            avg_hrv=('hrv', 'mean'),
            avg_sleep=('sleep_quality', 'mean')
        ).reset_index()

        profiles['risk_score'] = (profiles['avg_stress'] * 0.45) + ((100 - profiles['avg_hrv']) * 0.35) + ((100 - profiles['avg_sleep']) * 0.20)
        profiles = profiles.sort_values('risk_score', ascending=False).reset_index(drop=True)

        top_profiles = profiles.head(20).copy()

        bar_colors = []
        for score in top_profiles['risk_score']:
            if score > 45:
                bar_colors.append(colors["red"])
            elif score > 30:
                bar_colors.append(colors["yellow"])
            else:
                bar_colors.append(colors["green"])

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        bars = ax2.bar(top_profiles['user'], top_profiles['risk_score'], color=bar_colors)
        
        ax2.set_ylabel('Risk Index (0-100)', fontsize=10)
        ax2.set_ylim(0, max(top_profiles['risk_score']) + 20)
        ax2.set_xticklabels(top_profiles['user'], rotation=45, ha='right', fontsize=7)
        st.pyplot(fig2)

    with colD:
        st.subheader("Scientific Correlation: Sleep vs HRV (All Team)")
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['sleep_quality'], df['hrv'])

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.regplot(x=df['sleep_quality'], y=df['hrv'], color=colors["navy"],
                    scatter_kws={'alpha':0.3, 's': 30, 'color': colors["orange"]},
                    line_kws={'label': f'Scientific Correlation ($R^2$={r_value**2:.2f})'}, ax=ax3)
        
        ax3.set_xlabel('Reported Sleep Quality (%)', fontsize=10)
        ax3.set_ylabel('HRV (ms)', fontsize=10)
        ax3.legend(fontsize=8)
        st.pyplot(fig3)

    st.markdown("<p style='text-align: center; color: gray;'>OptiSafe MVP v4.0 - Proprietary Dashboard</p>", unsafe_allow_html=True)


# =====================================================================
# APP ROUTING (THE "TRAFFIC COP")
# =====================================================================
if not st.session_state['logged_in']:
    login()
else:
    run_dashboard()
