from sensor_simulation import generate_sensor_data
from som_model import train_som
from visualization import plot_clusters
import numpy as np
import pickle  # add this import at the top

# Generate robot sensor data
data = generate_sensor_data()

# Train SOM model
som = train_som(data)

# Visualize clusters
plot_clusters(som, data)

# Simple anomaly detection
threshold = 0.5
anomaly_count = 0
for x in data:
    w = som.winner(x)
    dist = np.linalg.norm(x - som.get_weights()[w[0]][w[1]])
    if dist > threshold:
        anomaly_count += 1

print(f"Detected anomalies: {anomaly_count}")

# Save trained SOM model for later use
with open('som_model.pkl', 'wb') as f:
    pickle.dump(som, f)
print("Trained SOM model saved as som_model.pkl")