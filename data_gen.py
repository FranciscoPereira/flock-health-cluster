"""
Synthetic data generator: 3 NHS nodes with respiratory syndrome ED visits.
Data stays LOCAL - never leaves each node.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_data(output_dir: str = "data"):
    """
    Generate 60 days of ED visit data for 3 NHS nodes.
    - Normal seasonal pattern
    - Node B has outbreak spike (doubling) on days 45-52
    """
    Path(output_dir).mkdir(exist_ok=True)
    np.random.seed(42)

    days = 60
    baseline_visits = 45

    # Create base pattern (seasonal variation + noise)
    t = np.arange(days)
    seasonal = 10 * np.sin(2 * np.pi * t / 30)  # Monthly pattern
    noise = np.random.normal(0, 3, days)

    data_dict = {}

    # Node A: Normal pattern
    node_a_visits = baseline_visits + seasonal + noise
    node_a_visits = np.maximum(node_a_visits, 10)  # Min 10 visits
    data_dict["node_a"] = node_a_visits.astype(int)

    # Node B: Normal pattern + outbreak spike on days 45-52
    node_b_visits = baseline_visits + seasonal + noise
    node_b_visits[45:53] *= 2  # OUTBREAK: double visits
    node_b_visits = np.maximum(node_b_visits, 10)
    data_dict["node_b"] = node_b_visits.astype(int)

    # Node C: Normal pattern (different seed for variety)
    np.random.seed(43)
    noise_c = np.random.normal(0, 3, days)
    node_c_visits = baseline_visits + seasonal + noise_c
    node_c_visits = np.maximum(node_c_visits, 10)
    data_dict["node_c"] = node_c_visits.astype(int)

    # Save as CSVs (local to each node)
    for node_name, visits in data_dict.items():
        df = pd.DataFrame({
            "day": np.arange(1, days + 1),
            "respiratory_visits": visits,
            "node": node_name.upper()
        })
        filepath = Path(output_dir) / f"{node_name}.csv"
        df.to_csv(filepath, index=False)
        print(f"✓ Generated {node_name}.csv: {len(df)} days (stays local at {node_name})")
        if node_name == "node_b":
            print(f"  └─ OUTBREAK INJECTED: days 45-52 (visits ~{visits[45:53].mean():.0f}/day)")

    return data_dict

if __name__ == "__main__":
    data_dict = generate_synthetic_data()
    print("\n✓ Data generation complete. Each node has its own CSV (data never moves).")
