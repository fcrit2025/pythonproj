import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
def create_sample_data():
    x = np.linspace(0, 50, 500)
    y = np.sin(x) + np.random.normal(0, 0.2, 500)
    return y

# Prepare sequences
def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Main execution
def main():
    # Create and prepare data
    data = create_sample_data()
    X, y = create_sequences(data)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Add feature dimension
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build model
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Plot results
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()