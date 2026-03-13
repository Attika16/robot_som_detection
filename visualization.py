import matplotlib.pyplot as plt

def plot_clusters(som, data, normal_count=500):
    plt.figure(figsize=(6,6))

    # First part: normal data
    for x in data[:normal_count]:
        w = som.winner(x)
        plt.plot(w[0], w[1], 'bo')  # blue dots

    # Second part: anomalies
    for x in data[normal_count:]:
        w = som.winner(x)
        plt.plot(w[0], w[1], 'ro')  # red dots

    plt.title("Robot Sensor Clusters (Blue=Normal, Red=Anomaly)")
    plt.xlabel("SOM X")
    plt.ylabel("SOM Y")
    plt.tight_layout()
    plt.savefig("cluster_plot.png")
    print("Plot saved as cluster_plot.png")