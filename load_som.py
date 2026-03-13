import pickle

# Load the saved SOM model
with open('som_model.pkl', 'rb') as f:
    som = pickle.load(f)

# Print basic info about the SOM
print("SOM weights shape:", som.get_weights().shape)
print("SOM trained successfully!")