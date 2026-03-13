# Robot Sensor Anomaly Detection using SOM

This project simulates robot sensor data and detects anomalies using a Self-Organizing Map (SOM).  
It demonstrates a **data science + robotics research project**, suitable for autonomous system analysis.

---

## Features
- Simulate robot sensor readings (normal + anomalies)
- Cluster data using Self-Organizing Map
- Detect anomalies automatically
- Visualize clusters in plots
- Save and reuse trained SOM model for new data

---

## Files
- `main.py` → Generate data, train SOM, visualize clusters
- `sensor_simulation.py` → Functions to simulate sensor readings
- `som_model.py` → Train SOM model
- `visualization.py` → Plot clusters
- `use_saved_som.py` → Load saved SOM and detect anomalies on new data
- `cluster_plot.png` → Plot from `main.py`
- `new_cluster_plot.png` → Plot from `use_saved_som.py`
- `som_model.pkl` → Saved SOM model

---

## How to Run

1. Install dependencies:

```bash
pip install numpy matplotlib scikit-learn minisom