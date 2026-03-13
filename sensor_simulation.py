import numpy as np

def generate_sensor_data(samples=500):

    # normal robot sensor readings
    normal_data = np.random.normal(loc=0.5, scale=0.1, size=(samples,3))

    # abnormal sensor readings (anomalies)
    anomaly = np.random.normal(loc=1.2, scale=0.2, size=(20,3))

    # combine both datasets
    data = np.vstack((normal_data, anomaly))

    return data