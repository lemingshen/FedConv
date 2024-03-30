# FedConv: Learning on Model for Heterogeneous Federated Clients
This is the Python implementation for MobiSys 2024 paper: "FedConv: Learning on Models for Heterogeneous Federated Clients". Our paper aims to address the challenges brought by heteogeneous models in federated learning:

<img src=images/scenario.png width=50%/>

## Requreiments
- Hardware
    - A server with GPT
    - Multiple clients (edge devices)
- Software
    - Operating System: Ubuntu 22.04 LTS
    - Python 3.10.12
    - PyTorch 1.13.1+cu117
    - Flower 1.7.0

## FedConv Overview

![System overview of FedConv](images/system.png)

- Server
    - Initialze a large global model.
    - Apply _Convolutional Compression_ on the large global and generate heterogeneous sub-models for clients.
    - Apply _Transposed Convolutional Dilation_ on the received hetergeneous client models to transform them into large models.
    - Apply _Weighted Average Aggregation_ on the rescaled large models and perform model aggregation for the next global communiction round.

- Heterogeneous Clients
    - Perform their resource profiles to determine a set of shrinkage ratios and transmit the shrinkage ratios to the server.
    - Perform local training as in traditional FL.

## Project Structure
```
|-- datasets                    // datasets used for evaluation
    |-- Image
        |-- MNIST
        |-- CIFAR10
        |-- CINIC10
    |-- HAR
        |-- WiAR
        |-- Depth_Camera
        |-- HARBox

|-- network_architecture        // models for different datasets
    |-- MNIST.py
    |-- CIFAR.py
    |-- CINIC.py
    |-- WiAR.py
    |-- Depth_Camera.py
    |-- HARBox.py

|-- Results                     // evaluation results
    |-- evaluation              // includes global model accuracy and client model accuracy
    |-- memory                  // includes CPU & GPU memeory usage, network usage, and wall-clock time
    |-- saved_models            // saved state_dict
    |-- weight_vector           // saved weight vectors

|-- utils                       // FedConv settings
    |-- client_settings.py      // client-side model related settings
    |-- data_processing.py      // data loading & processing related settings
    |-- federated settings.py   // convolutional compression, TC dilation, and weighted aggregation related settings

|-- Replace                     // modified PyTorch packages
    |-- batchnorm.py
    |-- conv.py
    |-- init.py
    |-- linear.py
    |-- module.py

|-- images                      // figures used in this README.md

|-- state_dict                  // temporary folder for saving intermediate state_dict

|-- config.py                   // configuration for hyper-parameters of FedConv
|-- server.py                   // run server
|-- client.py                   // run client
|-- strategy.py                 // FL strategies: FedConv
|-- README.md
|-- requirements.txt
```

## Quick Start
### 1. Installtion
```bash
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
```

### 2. PyTorch Package Modification
Replace the following Python files from your local Python environment with the files provided in the `Replace` folder:

```
/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py
/usr/local/lib/python3.10/dist-packages/torch/nn/modules/conv.py
/usr/local/lib/python3.10/dist-packages/torch/nn/modules/batchnorm.py
/usr/local/lib/python3.10/dist-packages/torch/nn/modules/linear.py
/usr/local/lib/python3.10/dist-packages/torch/nn/init.py
```

Note that the paths to the above files may vary based on your local Python environment.

### 3. Server Configuration
```bash
python3 server.py
```

### 4. Client Configuration
```bash
python3 client.py --client_id <client_id>
```

## Notes
- Feel free to modify the hyper-parameters in the `config.py` and good luck.
- Please don't hesitate to reach out if you have any questions.