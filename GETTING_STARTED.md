# 🚀 Getting Started - 5 Minute Demo

## What You'll See

A complete federated learning demo where:
- **3 NHS nodes** train a shared disease detection model
- **Raw patient data never leaves any node**
- **Node B's outbreak** (injected days 45-52) is detected automatically
- **Nodes A & C** remain in normal status

---

## Installation (1 minute)

```bash
# Navigate to project
cd /Users/franciscopereira/flock-health-cluster

# Install dependencies (first time only)
python3 -m pip install -r requirements.txt
```

---

## Run Demo (2 minutes)

### Option 1: One Command (Recommended)
```bash
./run_demo.sh
```

**Output:** Shows complete pipeline:
1. ✓ Generated 3 node datasets
2. ✓ Federated model trained (2 rounds)
3. ✓ Anomaly detection complete
4. ✓ Results displayed

### Option 2: Step-by-Step
```bash
# 1. Generate data
python3 data_gen.py

# 2. Train federated model
python3 federated_train.py

# 3. Detect anomalies
python3 detect.py
```

### Option 3: Interactive Dashboard
```bash
streamlit run streamlit_app.py
```
- Opens browser at `http://localhost:8501`
- Click "Run Full Pipeline" button
- Watch real-time training and detection
- Explore interactive charts and alerts

---

## Expected Output

### Console Output
```
✓ Generated node_a.csv: 60 days (stays local at node_a)
✓ Generated node_b.csv: 60 days (stays local at node_b)
  └─ OUTBREAK INJECTED: days 45-52 (visits ~76/day)
✓ Generated node_c.csv: 60 days (stays local at node_c)

--- FEDERATED ROUND 1/2 ---
→ node_a: Training locally...
  └─ Training complete | Final loss: 0.0997
  → node_a: Weights exported (no data leaked)
...

======================================================================
ANOMALY DETECTION: Each node scans locally for outbreaks
======================================================================

→ NODE_A: Scanning for anomalies...
  Status: 🟢 NORMAL
  Anomalies detected: 0

→ NODE_B: Scanning for anomalies...
  Status: 🔴 ALERT
  Anomalies detected: 7
    └─ DAY 45: Error 0.2099 (Z-score: 5.66) 🚨

→ NODE_C: Scanning for anomalies...
  Status: 🟢 NORMAL
  Anomalies detected: 0
```

---

## Key Files

| File | Purpose |
|------|---------|
| `data_gen.py` | Generate 3 synthetic CSVs with respiratory ED visits |
| `autoencoder.py` | PyTorch autoencoder for anomaly detection |
| `federated_train.py` | Federated learning orchestration (FedAvg) |
| `detect.py` | Local anomaly detection on each node |
| `streamlit_app.py` | Interactive dashboard |
| `requirements.txt` | Python dependencies |

---

## Privacy Guarantees ✅

✓ **Raw patient data never leaves any node**
- Each node trains locally on its own data
- Only model weights (76 KB of numbers) are shared
- Aggregator receives 3 model weight sets, averages them
- Result: shared model that improves all nodes

✓ **Outbreak detection is local**
- Each node uses the global model on its own data
- Each node independently decides if there's an alert
- Each node controls what it reports

✓ **Audit trail**
- Every alert has timestamp and node ID
- Z-scores show statistical confidence
- Full reproducibility for compliance

---

## Troubleshooting

**Q: `ModuleNotFoundError: No module named 'torch'`**
```bash
python3 -m pip install torch numpy pandas streamlit
```

**Q: `streamlit: command not found`**
```bash
python3 -m streamlit run streamlit_app.py
```

**Q: No alerts detected?**
This is expected! The model learns the normal pattern (days 1-44). If you see 0 alerts, it means the anomaly threshold needs adjustment. Try changing `sigma_threshold` in `detect.py` from `2.5` to `2.0`.

**Q: Want to see raw data?**
```bash
head data/node_b.csv  # Shows days 1-10
tail data/node_b.csv  # Shows days 50-60 (outbreak period)
```

---

## Next Steps

1. **Understand the code** - Read comments in each `.py` file
2. **Experiment** - Try different thresholds, training rounds, model sizes
3. **Deploy** - Connect to real Flock.io infrastructure for multi-institution use
4. **Extend** - Add differential privacy, LSTM models, streaming data

---

## Documentation

- **README.md** - Full project description
- **PROJECT_SUMMARY.md** - Technical deep dive
- **This file** - Quick start guide

---

## One-Liner Commands

```bash
# Install and run everything
python3 -m pip install -q -r requirements.txt && ./run_demo.sh

# Run just training
python3 data_gen.py && python3 federated_train.py

# Run just detection
python3 detect.py

# Launch dashboard
python3 -m streamlit run streamlit_app.py
```

---

## Questions?

Check `README.md` for detailed documentation and references.

**Built with:** Python 3.10 + PyTorch + Flock.io SDK

**Status:** ✅ Tested and working end-to-end
