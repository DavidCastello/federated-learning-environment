import numpy as np
import tensorflow as tf
from flwr.common import FitRes, EvaluateRes
import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train the model using the provided parameters."""
        # Set the model parameters
        self.model.set_weights(parameters)

        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=16, verbose=2)

        # Save the model in h5 format with a unique name
        #timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #self.model.save('my_model_{}.h5'.format(timestamp))

        # Return the updated model parameters
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)
        
        #loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        num_examples_test = len(self.x_test)
        
        return loss, num_examples_test, {"accuracy": accuracy}