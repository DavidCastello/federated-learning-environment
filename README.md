### Federated Learning environment 
Audio Classification

## Run a local simulation
This is an experimental project. It runs a server and N=6 clients within a local machine.

To efficiently conduct the simulations, make sure to enable multiprocessing in your machine. Each client will run in parallel.

All clients will initalize the model defined in `model.py` with random weights.

###### With conda env
- To create conda environment: 
```bash
conda create --name fl-env python=3.10
conda activate fl-env 
```

- To install requirements:
```bash
pip install -r requirements.txt
```

### Simulation

###### Start the server
- Start the server with the predifined strategy (Federated Averaging in this case):
```bash
python3 server.py
```


###### Prepare data

- Start clients with locally defined data partitions. Run in a different terminal the following command: 
```bash
python3 main.py
```


### Notes: 
- To test non-IID data scenarios, use `main-noniid.py` with a custom data distribution algorithm. It is prepared to use a public dataset to balance non-IID data across clients.
 

Author: David Castelló Tejera



