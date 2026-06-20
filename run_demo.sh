#!/bin/bash
set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🏥 Federated Disease Cluster Detection - Full Demo             ║"
echo "║  3 NHS nodes, federated learning, outbreak detection           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Create directories
mkdir -p data models

# Step 1: Generate data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Generating synthetic data for 3 NHS nodes..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 data_gen.py
echo ""

# Step 2: Federated training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Running federated learning (2 rounds)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 federated_train.py
echo ""

# Step 3: Anomaly detection
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Running anomaly detection on all nodes..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 detect.py
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Demo complete!                                              ║"
echo "║                                                                 ║"
echo "║  📊 To view interactive dashboard, run:                        ║"
echo "║      streamlit run streamlit_app.py                            ║"
echo "║                                                                 ║"
echo "║  🔐 Key Properties:                                            ║"
echo "║    ✓ Raw data never left any node                             ║"
echo "║    ✓ Only model weights aggregated (FedAvg)                   ║"
echo "║    ✓ Node B detected outbreak via reconstruction error        ║"
echo "║    ✓ Nodes A & C remained in normal status                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
