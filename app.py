import pandas as pd
import streamlit as st

# 1. Leer el archivo CSV y guardarlo en un DataFrame llamado 'df'
df = pd.read_csv('wearable_data_full_day_bueno.csv')

# 2. Mostrar las primeras 5 filas para verificar que se cargó correctamente
df.head()

# Ver información general de las columnas, valores nulos y tipos de datos
df.info()

# Ver estadísticas descriptivas básicas (promedios, mínimos, máximos de cada columna)
df.describe()

!pip install streamlit
import streamlit as st
st.title("OptiSafe Dashboard")
st.sidebar.header("Supervisor Controls")

# Create a dropdown menu to select the worker
selected_user = st.sidebar.selectbox("Select Employee", df['user'].unique())

# Show high-level metrics
col1, col2, col3 = st.columns(3)
col1.metric("Active Workers", len(df['user'].unique()))
col2.metric("Alerts Today", "3") # You can calculate this dynamically
col3.metric("Pieces Rescued", recovered_pieces)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=== STARTING OPTISAFE MVP V4.0 PROCESSING ENGINE (WITH REAL ROI) ===\n")

# ---------------------------------------------------------
# VISUAL CONFIGURATION AND DATA LOADING
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")
colors = {"navy": "#1A2A3A", "orange": "#F26D21", "gray": "#B0B0B0",
          "green": "#28a745", "yellow": "#ffc107", "red": "#dc3545"}

# 1. Read the real CSV file
try:
    df = pd.read_csv('wearable_data_full_day_bueno.csv')
    print("File 'wearable_data_full_day_bueno.csv' loaded successfully.")
except FileNotFoundError:
    print("ERROR: Make sure to upload the file 'wearable_data_full_day_bueno.csv' to Google Colab.")
    raise

# Clean column names just in case (remove trailing spaces)
df.columns = df.columns.str.strip()

# Convert timestamp to a manageable format (and avoid ValueError with format='mixed')
df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='mixed')
df['hour_numeric'] = df['timestamp_dt'].dt.hour + df['timestamp_dt'].dt.minute/60.0

# EXTRACT ONLY THE WORK SHIFT: Filter hours where pieces were actually produced
df_work = df[df['pieces_assembled_hour'] > 0].copy()

# ---------------------------------------------------------
# 1. ROBUST ALERTS (DYNAMIC FATIGUE THRESHOLD)
# ---------------------------------------------------------
print("\n1. Generating predictive alert system...")

# Select the employee with the highest stress level for the example
worst_user = df_work.groupby('user')['stress_level'].max().idxmax()
df_user = df_work[df_work['user'] == worst_user].copy()

# Moving average and statistical thresholds
df_user['stress_trend'] = df_user['stress_level'].rolling(window=2, min_periods=1).mean()
stress_mean = df_user['stress_level'].mean()
stress_std = df_user['stress_level'].std()

# The dynamic threshold is set to +1 standard deviation
threshold = stress_mean + stress_std
df_user['alert'] = df_user['stress_level'] > threshold

plt.figure(figsize=(10, 5))
plt.plot(df_user['hour_numeric'], df_user['stress_level'], color=colors["gray"], alpha=0.5, label='Raw Stress')
plt.plot(df_user['hour_numeric'], df_user['stress_trend'], color=colors["navy"], linewidth=3, label='Accumulated Trend')

plt.axhline(threshold, color=colors["orange"], linestyle='--', linewidth=2, label='Critical Intervention Threshold')
plt.fill_between(df_user['hour_numeric'], threshold, df_user['stress_level'].max() + 5, color=colors["orange"], alpha=0.1)

# Mark the alerts
alerts = df_user[df_user['alert']]
plt.scatter(alerts['hour_numeric'], alerts['stress_trend'], color=colors["red"], s=150, zorder=5, label='Alert: Break Recommended')

plt.title(f'Real-Time Biometric Monitor: {worst_user}', fontsize=14, fontweight='bold')
plt.xlabel('Hour of the Day (24h Format)', fontsize=12)
plt.ylabel('Accumulated Stress Level', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
st.pyplot(plt.gcf())

# ---------------------------------------------------------
# 2. RISK DASHBOARD: ADVANCED COMPOSITE INDEX
# ---------------------------------------------------------
print("2. Calculating Operational Fatigue Risk Score (FRS)...")

# Group by user during their working hours
profiles = df_work.groupby('user').agg(
    avg_stress=('stress_level', 'mean'),
    avg_hrv=('hrv', 'mean'),
    avg_sleep=('sleep_quality', 'mean')
).reset_index()

# FRS (Fatigue Risk Score) formula adapted to real data
# Penalizes high stress, low HRV, and poor sleep
profiles['risk_score'] = (profiles['avg_stress'] * 0.45) + ((100 - profiles['avg_hrv']) * 0.35) + ((100 - profiles['avg_sleep']) * 0.20)

# Sort from lowest to highest risk
profiles = profiles.sort_values('risk_score').reset_index(drop=True)

# Assign traffic light colors based on position
num_users = len(profiles)
bar_colors = []
for i in range(num_users):
    if i < num_users / 3:
        bar_colors.append(colors["green"])
    elif i < (num_users / 3) * 2:
        bar_colors.append(colors["yellow"])
    else:
        bar_colors.append(colors["red"])

plt.figure(figsize=(10, 5))
bars = plt.bar(profiles['user'], profiles['risk_score'], color=bar_colors)

for bar, score, hrv in zip(bars, profiles['risk_score'], profiles['avg_hrv']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'FRS: {score:.1f}\nHRV: {hrv:.1f}',
             ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)

plt.title('Team Dashboard: Risk Traffic Light (FRS)', fontsize=14, fontweight='bold')
plt.ylabel('Risk Index (0-100)', fontsize=12)
plt.ylim(0, max(profiles['risk_score']) + 20)
plt.tight_layout()
st.pyplot(plt.gcf())

# ---------------------------------------------------------
# 3. CORRELATION: SLEEP VS HRV
# ---------------------------------------------------------
print("3. Validating clinical correlation (Sleep Quality vs HRV Recovery)...")
# We use the full DF (including sleep hours) to see overall quality
slope, intercept, r_value, p_value, std_err = stats.linregress(df['sleep_quality'], df['hrv'])

plt.figure(figsize=(10, 5))
sns.regplot(x=df['sleep_quality'], y=df['hrv'], color=colors["navy"],
            scatter_kws={'alpha':0.5, 's': 80, 'color': colors["orange"]},
            line_kws={'label': f'Scientific Correlation ($R^2$={r_value**2:.2f})'})

plt.title('Impact of Sleep Quality on Heart Rate Variability (HRV)', fontsize=14, fontweight='bold')
plt.xlabel('Reported Sleep Quality (%)', fontsize=12)
plt.ylabel('HRV (ms)', fontsize=12)
plt.legend()
plt.tight_layout()
st.pyplot(plt.gcf())

# ---------------------------------------------------------
# 4. PRODUCTIVITY SIMULATION & ROI (ASSEMBLED PIECES)

print("4. Generating Return on Investment Simulation (Impact on Produced Pieces)...")

# We take the actual pieces of the worst user as our "Reactive" scenario (Status Quo)
actual_pieces = df_user['pieces_assembled_hour'].values
shift_hours = np.arange(1, len(actual_pieces) + 1)
actual_stress = df_user['stress_level'].values

optisafe_pieces = []
alert_triggered = False
recovered_pieces = 0

for i in range(len(actual_pieces)):
    current_piece = actual_pieces[i]
    stress = actual_stress[i]

    # If OptiSafe triggered an alert in previous hours, the employee rested and the drop in pieces is mitigated
    if alert_triggered:
        # We simulate that thanks to the break, the operator maintains a productive rhythm
        # higher than the real extreme fatigue scenario (e.g., rescues between 2 to 4 pieces per hour)
        rescue = int(np.random.uniform(2, 5))
        recovered_pieces += rescue
        optisafe_pieces.append(current_piece + rescue)
    else:
        optisafe_pieces.append(current_piece)

    # If stress exceeds the threshold, the OptiSafe protocol is activated for the following hours
    if stress > threshold:
        alert_triggered = True

plt.figure(figsize=(10, 5))
plt.plot(shift_hours, optisafe_pieces, marker='o', color=colors["navy"], linewidth=3, label='With OptiSafe (Proactive Intervention)')
plt.plot(shift_hours, actual_pieces, marker='X', color=colors["orange"], linestyle='--', linewidth=2, label='Status Quo (Drop due to Real Fatigue)')

plt.fill_between(shift_hours, optisafe_pieces, actual_pieces, color=colors["green"], alpha=0.15,
                 label=f'ROI: +{recovered_pieces} Extra Pieces per Shift')

plt.title(f'Daily Return on Investment: Productivity Maintenance ({worst_user})', fontsize=14, fontweight='bold')
plt.xlabel('Elapsed Shift Hour', fontsize=12)
plt.ylabel('Pieces Assembled per Hour', fontsize=12)
plt.legend(loc='lower left')
plt.tight_layout()
st.pyplot(plt.gcf())

print("\n=== OPTISAFE MVP V4.0 COMPLETED SUCCESSFULLY ===")
