#!/usr/bin/env python
"""
Create a simple LSTM model and scaler for testing purposes.

This script creates a basic LSTM model and scaler and saves them to the model_artifacts directory.
This is for TESTING ONLY and should not be used in production.
"""

import os
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Create directory if it doesn't exist
os.makedirs('model_artifacts', exist_ok=True)

# Create a simple LSTM model
def create_test_model():
    # Define model parameters
    sequence_length = 10
    features = 1
    
    # Create model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create some dummy data to train the model
    X_train = np.random.random((100, sequence_length, features))
    y_train = np.random.randint(0, 2, size=(100,))
    
    # Train the model for a few epochs
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)
    
    # Save the model
    model.save('model_artifacts/lstm_model.h5')
    print("Model saved to model_artifacts/lstm_model.h5")
    
    # Create and save a scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    sample_data = np.random.random((100, 1)) * 10000  # Random price-like data
    scaler.fit(sample_data)
    
    with open('model_artifacts/lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to model_artifacts/lstm_scaler.pkl")
    
    return model, scaler

if __name__ == "__main__":
    print("Creating test LSTM model and scaler...")
    model, scaler = create_test_model()
    
    # Test the model with a sample input
    test_input = np.random.random((1, 10, 1))
    prediction = model.predict(test_input)[0][0]
    print(f"Test prediction: {prediction:.4f}")
    
    print("Done! You can now run the bot with these test model files.")
    print("WARNING: This is for testing purposes only and should not be used in production.")
