import flwr as fl
from flwr.server.strategy import FedAvg

if __name__ == "__main__":
    print("---------SERVER HAS STARTED!-------------")

    # Use default aggregation strategy
    strategy = FedAvg()

    # Create server configuration
    config = fl.server.ServerConfig(num_rounds=20)
    
    # Start server
    fl.server.start_server(server_address="localhost:8080", config=config, strategy=strategy)