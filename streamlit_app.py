"""
Streamlit dashboard: Real-time anomaly detection map + alerts + HUD.
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from data_gen import generate_synthetic_data
from federated_train import federated_training_loop
from detect import run_anomaly_detection
from autoencoder import get_device

# Page config
st.set_page_config(
    page_title="NHS Federated Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .hud-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: 2px solid #00ff00;
        border-radius: 10px;
        padding: 20px;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
    }
    .alert-box {
        background: #8b0000;
        border: 2px solid #ff0000;
        border-radius: 8px;
        padding: 15px;
        color: #ffffff;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
    .normal-box {
        background: #004d00;
        border: 2px solid #00ff00;
        border-radius: 8px;
        padding: 15px;
        color: #00ff00;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: CONTROL PANEL
# ============================================================================
st.sidebar.header("⚙️ Federated Learning Control")

if st.sidebar.button("🚀 Run Full Pipeline", key="run_pipeline"):
    st.session_state.pipeline_running = True

if st.sidebar.button("🔄 Reset", key="reset"):
    st.session_state.clear()

# ============================================================================
# MAIN DASHBOARD
# ============================================================================
st.title("🏥 NHS Federated Disease Cluster Detection")
st.markdown("*Collaborative outbreak detection without sharing patient data*")

# ============================================================================
# HUD: System Status
# ============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="hud-box">
        <div style="font-size: 18px; font-weight: bold;">SYSTEM STATUS</div>
        <div style="font-size: 14px; margin-top: 10px;">
            Federated Model: <span style="color: #00ff00;">INITIALIZED</span><br>
            Nodes Active: <span style="color: #00ff00;">3/3</span><br>
            Data Sharing: <span style="color: #ff0000;">DISABLED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="hud-box">
        <div style="font-size: 18px; font-weight: bold;">ARCHITECTURE</div>
        <div style="font-size: 14px; margin-top: 10px;">
            Learning: <span style="color: #00ff00;">FedAvg</span><br>
            Model: <span style="color: #00ff00;">Autoencoder</span><br>
            Transport: <span style="color: #00ff00;">Weights Only</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="hud-box">
        <div style="font-size: 18px; font-weight: bold;">LAST RUN</div>
        <div style="font-size: 14px; margin-top: 10px;">
            Time: <span style="color: #ffff00;">--:--</span><br>
            Model Version: <span style="color: #00ff00;">v1.0-FL</span><br>
            Accuracy: <span style="color: #00ff00;">N/A</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# RUN PIPELINE
# ============================================================================
if st.session_state.get("pipeline_running", False):
    st.info("🔄 Running federated learning pipeline...")

    # Step 1: Generate data
    st.markdown("### Step 1️⃣ : Generate Synthetic Data")
    with st.spinner("Generating 60 days of respiratory ED visit data..."):
        generate_synthetic_data("data")
    st.success("✓ Generated 3 node datasets (local, no sharing)")

    # Step 2: Federated training
    st.markdown("### Step 2️⃣ : Federated Training (FedAvg)")
    with st.spinner("Training autoencoder across 3 nodes..."):
        Path("models").mkdir(exist_ok=True)
        device = get_device()
        global_model_weights = federated_training_loop(rounds=2)
        with open("models/global_model.pt", "wb") as f:
            f.write(global_model_weights)
    st.success("✓ Federated model trained and aggregated")

    # Step 3: Anomaly detection
    st.markdown("### Step 3️⃣ : Detect Anomalies Locally")
    with st.spinner("Running anomaly detection on each node..."):
        detection_results = run_anomaly_detection(global_model_weights)

    st.success("✓ Anomaly detection complete")

    # ========================================================================
    # RESULTS: NODE STATUS MAP
    # ========================================================================
    st.markdown("---")
    st.markdown("## 📍 Regional Status Map")

    col1, col2, col3 = st.columns(3)
    node_results = detection_results["node_results"]

    with col1:
        status = "🔴 ALERT" if node_results["node_a"]["is_alert"] else "🟢 NORMAL"
        st.markdown(f"""
        <div class="{'alert-box' if node_results['node_a']['is_alert'] else 'normal-box'}">
            <b>NODE A - Wales</b><br>
            Status: {status}<br>
            Anomalies: {node_results['node_a']['anomaly_count']}<br>
            Threshold: {node_results['node_a']['threshold']:.4f}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        status = "🔴 ALERT" if node_results["node_b"]["is_alert"] else "🟢 NORMAL"
        st.markdown(f"""
        <div class="{'alert-box' if node_results['node_b']['is_alert'] else 'normal-box'}">
            <b>NODE B - England</b><br>
            Status: {status}<br>
            Anomalies: {node_results['node_b']['anomaly_count']}<br>
            Threshold: {node_results['node_b']['threshold']:.4f}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status = "🔴 ALERT" if node_results["node_c"]["is_alert"] else "🟢 NORMAL"
        st.markdown(f"""
        <div class="{'alert-box' if node_results['node_c']['is_alert'] else 'normal-box'}">
            <b>NODE C - Scotland</b><br>
            Status: {status}<br>
            Anomalies: {node_results['node_c']['anomaly_count']}<br>
            Threshold: {node_results['node_c']['threshold']:.4f}
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # RESULTS: ANOMALY SCORE CHARTS
    # ========================================================================
    st.markdown("---")
    st.markdown("## 📊 Anomaly Score Progression")

    for node_name in ["node_a", "node_b", "node_c"]:
        result = node_results[node_name]
        errors = result["errors"]
        threshold = result["threshold"]

        df_chart = pd.DataFrame({
            "Window": range(len(errors)),
            "Reconstruction Error": errors,
            "Threshold": threshold,
            "Mean Error": result["mean_error"]
        })

        st.line_chart(df_chart.set_index("Window"), use_container_width=True)
        st.caption(f"**{node_name.upper()}** - Error progression (red = threshold, blue = actual)")

    # ========================================================================
    # RESULTS: ALERT LOG
    # ========================================================================
    st.markdown("---")
    st.markdown("## 🚨 Alert Log")

    alerts = detection_results["alerts"]
    if alerts:
        alert_df = pd.DataFrame(alerts)
        alert_df["timestamp"] = pd.to_datetime(alert_df["timestamp"]).dt.strftime("%H:%M:%S")
        st.dataframe(alert_df[["timestamp", "node", "day", "severity", "error", "z_score"]], use_container_width=True)
    else:
        st.info("✓ No alerts detected")

    # ========================================================================
    # KEY PROPERTIES
    # ========================================================================
    st.markdown("---")
    st.markdown("## 🔐 Security & Privacy Assurance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ✅ **Data Minimization**
        - Raw patient data never left any node
        - Only model weights aggregated
        - Each node computes locally

        ✅ **Federated Learning**
        - FedAvg: Average 3 local models
        - No central data collection
        - Privacy-by-design
        """)

    with col2:
        st.markdown("""
        ✅ **Anomaly Detection**
        - Reconstruction error > 3-sigma = alert
        - No training data disclosure
        - Independent node decision-making

        ✅ **Audit Trail**
        - Each alert logged with timestamp
        - Node identity preserved
        - Severity classification included
        """)

    st.session_state.pipeline_running = False

else:
    st.markdown("""
    ### Welcome to the Federated Disease Detection Demo

    **Click the "Run Full Pipeline" button to:**
    1. Generate synthetic respiratory ED visit data for 3 NHS nodes
    2. Collaboratively train an anomaly detection model via federated learning
    3. Detect outbreak clusters locally without sharing raw patient data

    **Key Features:**
    - 🔐 Privacy-Preserving: Raw data never leaves any node
    - 🤝 Federated: Only model weights aggregated
    - 🚨 Real-Time: Each node detects anomalies independently
    - 📊 Transparent: Full audit trail of alerts and decisions
    """)
