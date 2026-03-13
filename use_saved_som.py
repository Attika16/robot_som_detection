import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the saved SOM
with open('som_model.pkl', 'rb') as f:
    som = pickle.load(f)

print("SOM model loaded successfully!")

# Generate new robot sensor data
# Normal + anomalies
normal_data = np.random.normal(loc=0.5, scale=0.1, size=(500,3))
anomaly_data = np.random.normal(loc=1.2, scale=0.2, size=(20,3))
new_data = np.vstack((normal_data, anomaly_data))

# Simple anomaly detection
threshold = 0.5
detected_anomalies = 0
for x in new_data:
    w = som.winner(x)
    dist = np.linalg.norm(x - som.get_weights()[w[0]][w[1]])
    if dist > threshold:
        detected_anomalies += 1

print(f"Detected anomalies in new data: {detected_anomalies}")

# Visualize clusters
plt.figure(figsize=(6,6))
for x in new_data[:500]:
    w = som.winner(x)
    plt.plot(w[0], w[1], 'bo')  # blue = normal

for x in new_data[500:]:
    w = som.winner(x)
    plt.plot(w[0], w[1], 'ro')  # red = anomaly

plt.title("New Robot Sensor Data Clusters (Blue=Normal, Red=Anomaly)")
plt.xlabel("SOM X")
plt.ylabel("SOM Y")
plt.tight_layout()
plt.savefig("new_cluster_plot.png")
plt.show(block=True)