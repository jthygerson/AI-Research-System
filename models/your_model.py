# models/your_model.py

import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

def run_model(experiment_plan):
    # Parse the experiment_plan to extract parameters
    # For demonstration, we'll train a simple neural network on MNIST
    try:
        # Load dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255
        x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255

        # Build model
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train model
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }

        logging.info(f"Experiment results: {results}")
        return results

    except Exception as e:
        logging.error(f"Error in run_model: {e}")
        return None
