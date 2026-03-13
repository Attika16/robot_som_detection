from minisom import MiniSom

def train_som(data):

    som = MiniSom(10, 10, data.shape[1], sigma=1.0, learning_rate=0.5)

    som.random_weights_init(data)

    som.train_random(data, 1000)

    return som