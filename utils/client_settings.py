import math
import json
import time
import torch
import psutil
import config
import numpy as np
import torch.nn as nn

from network_architecture import *
from collections import OrderedDict
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader


# get a model's parameters in list format
def get_parameters(model) -> List[np.ndarray]:
    if config.use_FedBN:
        return [
            value.cpu().detach().numpy()
            for name, value in model.state_dict().items()
            if "bn" not in name
        ]
    else:
        return [
            value.cpu().detach().numpy()
            for _, value in model.state_dict().items()
        ]


# set parameters for model
def set_parameters(
        model,
        parameters: List[np.ndarray]
) -> None:
    if config.use_FedBN:
        keys = [key for key in model.state_dict().keys() if "bn" not in key]
        parameter_dictionary = zip(keys, parameters)
        state_dictionary = OrderedDict({
            key: torch.tensor(value)
            for key, value in parameter_dictionary
        })

        model.load_state_dict(state_dictionary, strict=False)
    else:
        parameter_dictionary = zip(model.state_dict().keys(), parameters)
        state_dictionary = OrderedDict({
            key: torch.tensor(value)
            for key, value in parameter_dictionary
        })

        model.load_state_dict(state_dictionary, strict=True)


def get_model(model_size_index):
    """
    generate the corresponding model according to config.py

    :param model_size_index: the model size
    :return: the corresponding model
    """

    torch.manual_seed(time.time_ns())
    shrink_ratio = math.pow(config.shrinkage_ratio, model_size_index)

    if config.dataset_type == "Image":
        if config.dataset_index == 0:
            model = MNIST.MNIST(shrink_ratio)
        elif config.dataset_index == 1:
            model = CIFAR.ResNet18(shrink_ratio)
        else:
            model = CINIC.GoogLeNet(shrink_ratio)
    else:
        if config.dataset_index == 0:
            model = WiAR.WiAR_CNN(shrink_ratio)
        elif config.dataset_index == 1:
            model = Depth_Camera.Depth_Camera_CNN(shrink_ratio)
        else:
            model = HARBox.HARBox_CNN(shrink_ratio)

    return model.to(config.device)


def timer(pid, save_dict, client_id):
    """
    the timer for client-side system resource monitoring

    :param pid: the pid of client process
    :param save_dict: dict to save
    :param client_id: client id
    :return: None
    """

    while True:
        time.sleep(1)
        real_time_memory_record(pid, save_dict, client_id)


def train_model(
        model,
        client_id: int,
        server_round: int,
        train_loader: DataLoader,
        epoch_number: int,
        example_number: Dict[str, int]
) -> Tuple[List[float], List[float], List[float]]:
    """
    train the client's model, i.e., local training

    :param client_id: the client id
    :param server_round: the current global communication round
    :param model: the client's model
    :param train_loader: client-side train dataset
    :param epoch_number: the number of training epochs
    :param example_number: the data number for this client
    :return: three list of average loss, accuracy and training time per sample during the local training
    """

    criterion = nn.CrossEntropyLoss()
    loss_list, accuracy_list, time_list = [], [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=config.client_learning_rate)

    print('Client id {} training . . .'.format(client_id))

    for epoch in range(epoch_number):
        print('Server round {} Client {} Local epoch {}'.format(
            server_round, client_id, epoch
        ))

        epoch_correct, epoch_total, epoch_loss = 0, 0, 0.0
        start_time = time.time()

        for train_data, train_label in train_loader:
            train_data = train_data.to(config.device)
            train_label = train_label.to(config.device)

            model_output = model(train_data)
            loss = criterion(model_output, train_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, train_predicted = torch.max(model_output.data, 1)

            epoch_loss += loss
            epoch_total += len(train_label)
            epoch_correct += (train_predicted == train_label).sum().item()

        end_time = time.time()
        time_per_sample = (end_time - start_time) / example_number["train_set"]
        average_loss = epoch_loss / example_number["train_set"]
        average_accuracy = epoch_correct / epoch_total * 100

        loss_list.append(float(average_loss))
        accuracy_list.append(average_accuracy)
        time_list.append(time_per_sample)

        print('Server round {} Client {} in Epoch {}, accuracy: {}%, average loss: {}'.format(
            server_round, client_id, epoch, average_accuracy, average_loss
        ))

    return (
        [loss for loss in loss_list],
        [accuracy for accuracy in accuracy_list],
        time_list
    )


def test_model(
        model,
        client_id: int,
        server_round: int,
        test_loader: DataLoader,
        example_number: Dict[str, int]
) -> Tuple[float, float]:
    """
    test the client model using the client-side test dataset

    :param model: the client's model
    :param client_id: the client id
    :param server_round: the current global communication round
    :param test_loader: the client-side test dataset
    :param example_number: the number of samples for the test dataset
    :return: average loss and accuracy
    """

    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    print('Server round {} Client {} testing . . .'.format(server_round, client_id))

    with torch.no_grad():
        for test_data, test_label in test_loader:
            test_data = test_data.to(config.device)
            test_label = test_label.to(config.device)

            model_output = model(test_data)

            _, test_predicted = torch.max(model_output.data, 1)

            loss += criterion(model_output, test_label).item()
            total += len(test_label)
            correct += (test_predicted == test_label).sum().item()

        average_loss = float(loss / example_number["test_set"])
        test_accuracy = correct / total * 100

    print('Server round {}, Client {}, test accuracy: {}%, average loss: {}'.format(
        server_round, client_id, test_accuracy, average_loss
    ))

    return average_loss, test_accuracy



def real_time_memory_record(pid, save_dict, client_id):
    """
    monitor the client's CPU and GPU memory usage during training

    save the data every 1 second

    :param pid: the pid of the current client's process
    :param save_dict: a dict to save
    :param client_id: the client's id
    :return: None
    """

    timestamp = int(time.time())
    process = psutil.Process(pid)

    # memory in MB
    save_dict[timestamp] = {
        "CPU": process.memory_info().rss / 1024 / 1024,
        "GPU": torch.cuda.memory_allocated() / 1024 / 1024
    }

    file_name = '{}/{}/cpu_gpu_memory_client{}.json'.format(
        config.memory_footprint_save_path,
        config.dataset_dict[config.dataset_type][config.dataset_index],
        client_id
    )

    with open(file_name, 'w') as memory_file:
        json.dump(save_dict, memory_file, indent=4)
