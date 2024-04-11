import json
import math
import torch
import config
import flwr as fl
import numpy as np
import torch.nn as nn

from tqdm import tqdm
import utils.client_settings
import utils.data_processing
from collections import OrderedDict
from flwr.common import Metrics, NDArrays
from typing import List, Dict, Tuple, Optional
from flwr.common import parameters_to_ndarrays


# modified Leaky ReLU (MLR) activation function
class Leaky_Decay_ReLU(nn.Module):
    def __init__(
        self,
        negative_slope=config.negative_slope,
        positive_slope=config.positive_slope,
        inplace=False,
    ):
        super(Leaky_Decay_ReLU, self).__init__()

        self.negative_slope = negative_slope
        self.positive_slope = positive_slope
        self.inplace = inplace

    def forward(self, input_value):
        if self.inplace:
            input_value = input_value.clone()

        negative_part = nn.functional.leaky_relu(
            input_value, negative_slope=self.negative_slope
        )
        positive = nn.functional.relu(input_value)
        positive_part = positive - self.positive_slope * nn.functional.relu(input_value)

        output = negative_part - positive_part
        return output


class Base_Filter(nn.Module):
    def __init__(
        self,
        kernel_size,
        conv_deconv=True,
        kernel_number=None,
        row_number=None,
        column_number=None,
    ):
        super(Base_Filter, self).__init__()

        self.kernel_number = kernel_number
        self.conv_deconv = conv_deconv
        self.row_number = row_number
        self.column_number = column_number

        # non convolution weight
        if kernel_number is None:
            if self.conv_deconv:
                self.projection = nn.Conv2d(1, 1, 1, 1, 0)
                self.compression = nn.utils.weight_norm(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        bias=False,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=0,
                    )
                )
            else:
                self.projection = nn.ConvTranspose2d(1, 1, 1, 1, 0)
                self.compression = nn.utils.weight_norm(
                    nn.ConvTranspose2d(
                        in_channels=1,
                        out_channels=1,
                        bias=False,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=0,
                    )
                )
        else:
            self.projection = nn.Sequential()
            self.compression = nn.Sequential()

            for kernel_index in range(kernel_number):
                if self.conv_deconv:
                    self.projection.append(nn.Conv2d(1, 1, 1, 1, 0))
                    self.compression.append(
                        nn.utils.weight_norm(
                            nn.Conv2d(
                                in_channels=1,
                                out_channels=1,
                                bias=False,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=0,
                            )
                        )
                    )
                else:
                    self.projection.append(nn.ConvTranspose2d(1, 1, 1, 1, 0))
                    self.compression.append(
                        nn.utils.weight_norm(
                            nn.ConvTranspose2d(
                                in_channels=1,
                                out_channels=1,
                                bias=False,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=0,
                            )
                        )
                    )

    def forward(self, x):
        # x shape (10, 1568), (10)
        if self.kernel_number is None:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(0).unsqueeze(0)

            x = self.projection(x)
            x = self.compression(x)

            if x.ndim == 2:
                x = x.squeeze(0)
            else:
                x = x.squeeze(0).squeeze(0)
        # x shape (32, 16, 3, 3)
        else:
            temp_output = []

            for index_x in range(self.row_number):
                temp_row = []

                for index_y in range(self.column_number):
                    kernel_index = index_x * self.column_number + index_y

                    current_kernel = x[:, :, index_x, index_y]
                    input_value = current_kernel.unsqueeze(0).float()

                    output = self.projection[kernel_index](input_value)
                    output = output + input_value
                    output = self.compression[kernel_index](output).squeeze(0)

                    temp_row.append(output)

                temp_output.append(temp_row)

            output = []

            for index_y in range(self.column_number):
                temp_column = []

                for index_x in range(self.row_number):
                    temp_column.append(temp_output[index_x][index_y])

                output.append(torch.stack(temp_column, dim=2))

            x = torch.stack(output, dim=3)

        x = Leaky_Decay_ReLU()(x)

        return x


def customized_fit_metric(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("fit metric", metrics)

    # multiply accuracy of each client by number of examples used
    accuracy_list = [
        example_number * metric["accuracy"] for example_number, metric in metrics
    ]
    example_list = [example_number for example_number, _ in metrics]

    # aggregate and return customized metric (weighted average)
    return {"accuracy": sum(accuracy_list) / sum(example_list)}


def customized_evaluate_metric(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("evaluate metric", metrics)

    # multiply accuracy of each client by number of examples used
    accuracy_list = [
        example_number * metric["accuracy"] for example_number, metric in metrics
    ]
    example_list = [example_number for example_number, _ in metrics]

    # aggregate and return customized metric (weighted average)
    return {"accuracy": sum(accuracy_list) / sum(example_list)}


# federated evaluation will be called by Flower after every communication round
def federated_evaluation(
    server_round: int,
    parameter: fl.common.NDArrays,
    configuration: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    """
    server-side model evaluation

        :param server_round: the current global communication round
        :param parameter: the global model's parameter
        :param configuration: configuration for evaluation
        :return: loss, accuracy
    """

    print(
        "=============== Server side evaluation, server round {} ===============".format(
            server_round
        )
    )

    model = utils.client_settings.get_model(model_size_index=0)
    (
        _,
        validation_loader,
        test_loader,
        example_number,
    ) = utils.data_processing.load_dataset(
        client_id=None,
        client_or_server=False,
        train_dataset=False,
        validation_dataset=True,
        test_dataset=True,
    )

    utils.client_settings.set_parameters(model, parameter)
    validation_accuracy, validation_loss = print_model_accuracy(
        model=model,
        dataset_name="validation dataset",
        data_loader=validation_loader,
        epoch=server_round,
    )
    test_accuracy, test_loss = print_model_accuracy(
        model=model,
        dataset_name="test dataset",
        data_loader=test_loader,
        epoch=server_round,
    )

    config.server_evaluation_dict[server_round] = {
        "validation_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
    }

    # save the server-side evaluation result
    with open(
        "{}/{}/server.json".format(
            config.evaluation_save_path,
            config.dataset_dict[config.dataset_type][config.dataset_index],
        ),
        "w",
    ) as server_evaluation_file:
        json.dump(config.server_evaluation_dict, server_evaluation_file, indent=4)

    # save server-side model state dict
    # if config.save_model and server_round % config.save_model_round_step == 0:
    #     # save server model
    #     model_file_path = "{}/{}/server_round_{}.pth".format(
    #         config.model_save_path,
    #         config.dataset_dict[config.dataset_type][config.dataset_index],
    #         server_round,
    #     )
    #     torch.save(model.state_dict(), model_file_path)

    # # save server confusion matrix
    # if config.save_confusion_matrix:
    #     confusion_matrix = utils.client_settings.get_confusion_matrix(
    #         model, test_loader
    #     )

    #     matrix_save_path = "{}/{}/server_round_{}_matrix.npy".format(
    #         config.confusion_matrix_save_path,
    #         config.dataset_dict[config.dataset_type][config.dataset_index],
    #         server_round,
    #     )
    #     np.save(matrix_save_path, confusion_matrix)

    return test_loss, {
        "validation_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
    }


def fit_config(server_round: int):
    """
    Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local epochs afterward.

        :param server_round: current global communication round
        :return: fit configuration
    """

    configuration = {
        "server_round": server_round,  # The current round of federated learning
        # "local_epochs": 1 if server_round < 2 else 2,  #
    }

    return configuration


def evaluate_config(server_round: int):
    """
    Return testing configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local epochs afterward.

        :param server_round: current global communication round
        :return: evaluate configuration
    """

    configuration = {
        "server_round": server_round,  # the current round index of federated learning
        "message": "something the server wants to say to the client",
    }

    return configuration


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """
    aggregate evaluation results obtained from multiple clients

        :param results: a list of example number and loss
        :return: aggregated loss
    """

    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]

    return sum(weighted_losses) / num_total_evaluation_examples


def initialize_conv_deconv_parameter(model_size_index, conv_deconv=True):
    """
    models with the same type have the same convolution parameters
    each client has unique transposed convolution parameters

    for each kernel or each weight parameter, assign different convolution or transposed convolution parameters

        :param model_size_index: the size of model
        :param conv_deconv: True for convolution parameters
        :return: corresponding state dict of the parameters
    """

    if model_size_index == 0:
        return None

    # -> Tuple[OrderedDict[str, torch.FloatTensor]]:
    current_shrinkage_ratio = math.pow(config.shrinkage_ratio, model_size_index)
    sample_model = utils.client_settings.get_model(model_size_index=0)

    result_state_dict = OrderedDict({})
    state_dict = sample_model.state_dict()

    weight_name_list = [key for key in state_dict.keys() if "weight" in key]
    bias_name_list = [key for key in state_dict.keys() if "bias" in key]

    input_weight_name = weight_name_list[0]
    output_weight_name = weight_name_list[-1]
    output_bias_name = bias_name_list[-1]

    for key, value in state_dict.items():
        if not value.requires_grad:
            result_state_dict[key] = None
            continue

        if "conv" in key:
            if "weight" in key:
                output_size = value.size(0)
                target_output_size = int(output_size * current_shrinkage_ratio)

                if value.ndim == 1:
                    conv_kernel_size_y = output_size - target_output_size + 1

                    temp_parameter = Base_Filter(
                        kernel_size=(1, conv_kernel_size_y), conv_deconv=conv_deconv
                    )
                else:
                    input_size = value.size(1)
                    kernel_x_size = value.size(2)
                    kernel_y_size = value.size(3)

                    target_input_size = int(input_size * current_shrinkage_ratio)

                    conv_kernel_size_x = output_size - target_output_size + 1
                    conv_kernel_size_y = input_size - target_input_size + 1

                    # shape example [16, 3, 7, 7]
                    if key == input_weight_name:
                        temp_parameter = Base_Filter(
                            kernel_size=(conv_kernel_size_x, 1),
                            kernel_number=(kernel_x_size * kernel_y_size),
                            row_number=kernel_x_size,
                            column_number=kernel_y_size,
                            conv_deconv=conv_deconv,
                        )
                    else:
                        temp_parameter = Base_Filter(
                            kernel_size=(conv_kernel_size_x, conv_kernel_size_y),
                            kernel_number=(kernel_x_size * kernel_y_size),
                            row_number=kernel_x_size,
                            column_number=kernel_y_size,
                            conv_deconv=conv_deconv,
                        )
            # the bias parameter of convolutional layers
            # shape example [32]
            else:
                output_size = value.size(0)
                target_output_size = int(output_size * current_shrinkage_ratio)
                conv_kernel_size_y = output_size - target_output_size + 1

                temp_parameter = Base_Filter(
                    kernel_size=(1, conv_kernel_size_y), conv_deconv=conv_deconv
                )
        elif "linear" in key:
            # the weight parameter of linear layer
            # shape example: [10, 1568]
            if "weight" in key:
                output_size = value.size(0)
                input_size = value.size(1)

                target_output_size = int(output_size * current_shrinkage_ratio)
                target_input_size = int(input_size * current_shrinkage_ratio)

                conv_kernel_size_x = output_size - target_output_size + 1
                conv_kernel_size_y = input_size - target_input_size + 1

                if key == output_weight_name:
                    temp_parameter = Base_Filter(
                        kernel_size=(1, conv_kernel_size_y), conv_deconv=conv_deconv
                    )
                else:
                    temp_parameter = Base_Filter(
                        kernel_size=(conv_kernel_size_x, conv_kernel_size_y),
                        conv_deconv=conv_deconv,
                    )
            # the bias parameter of linear layer
            # shape example: [10]
            else:
                output_size = value.size(0)
                target_output_size = int(output_size * current_shrinkage_ratio)
                kernel_y_size = output_size - target_output_size + 1

                if key == output_bias_name:
                    temp_parameter = Base_Filter(
                        kernel_size=(1, 1), conv_deconv=conv_deconv
                    )
                else:
                    temp_parameter = Base_Filter(
                        kernel_size=(1, kernel_y_size), conv_deconv=conv_deconv
                    )
        else:
            temp_parameter = None

        temp_parameter = (
            temp_parameter.to(config.device) if temp_parameter is not None else None
        )
        result_state_dict[key] = temp_parameter

    return result_state_dict


def initialize_weight_vector():
    """
    initialize the weight vectors for each layer and each client

        :return: weight vector state dict
    """

    sample_model = utils.client_settings.get_model(model_size_index=0)
    state_dict = sample_model.state_dict()
    result_state_dict = OrderedDict({})

    for key, value in state_dict.items():
        if value is None or not value.requires_grad or value.ndim == 0:
            result_state_dict[key] = None
        else:
            result_weight_list = []

            if value.ndim == 4:
                filter_number = value.size(0)
                channel_number = value.size(1)
                kernel_x = value.size(2)
                kernel_y = value.size(3)

                temp_weight_list = []

                for filter_index in range(filter_number):
                    filter_list = []
                    for channel_index in range(channel_number):
                        channel_list = []
                        for kernel_index_x in range(kernel_x):
                            x_list = []
                            for kernel_index_y in range(kernel_y):
                                current_weight = torch.randn(config.client_number)
                                x_list.append(current_weight)
                            channel_list.append(x_list)
                        filter_list.append(channel_list)
                    temp_weight_list.append(filter_list)

                for model_size_index in range(config.client_number):
                    current_weight = []

                    for filter_index in range(filter_number):
                        filter_list = []
                        for channel_index in range(channel_number):
                            channel_list = []
                            for kernel_index_x in range(kernel_x):
                                x_list = []
                                for kernel_index_y in range(kernel_y):
                                    x_list.append(
                                        temp_weight_list[filter_index][channel_index][
                                            kernel_index_x
                                        ][kernel_index_y][model_size_index].item()
                                    )
                                x_list = np.array(x_list)
                                channel_list.append(x_list)
                            channel_list = np.array(channel_list)
                            filter_list.append(channel_list)
                        filter_list = np.array(filter_list)
                        current_weight.append(filter_list)

                    current_weight = (
                        torch.from_numpy(np.array(current_weight))
                        .float()
                        .to(config.device)
                        .requires_grad_()
                    )
                    result_weight_list.append(current_weight)
            elif value.ndim == 2:
                filter_number = value.size(0)
                channel_number = value.size(1)

                temp_weight_list = []

                for filter_index in range(filter_number):
                    filter_list = []
                    for channel_index in range(channel_number):
                        # filter_list.append(torch.softmax(torch.randn(config.client_number), dim=0))
                        filter_list.append(torch.randn(config.client_number))
                    temp_weight_list.append(filter_list)

                for model_size_index in range(config.model_size_number):
                    current_weight = []

                    for filter_index in range(filter_number):
                        filter_list = []
                        for channel_index in range(channel_number):
                            filter_list.append(
                                temp_weight_list[filter_index][channel_index][
                                    model_size_index
                                ].item()
                            )
                        filter_list = np.array(filter_list)
                        current_weight.append(filter_list)

                    current_weight = (
                        torch.from_numpy(np.array(current_weight))
                        .float()
                        .to(config.device)
                        .requires_grad_()
                    )
                    result_weight_list.append(current_weight)
            else:
                filter_number = value.size(0)
                temp_weight_list = []

                for filter_index in range(filter_number):
                    temp_weight_list.append(torch.randn(config.client_number))

                for model_size_index in range(config.client_number):
                    current_weight = []

                    for filter_index in range(filter_number):
                        current_weight.append(
                            temp_weight_list[filter_index][model_size_index].item()
                        )

                    current_weight = (
                        torch.from_numpy(np.array(current_weight))
                        .float()
                        .to(config.device)
                        .requires_grad_()
                    )
                    result_weight_list.append(current_weight)

            result_state_dict[key] = result_weight_list

    return result_state_dict


def convolution(large_state_dict, conv_filter, shrinkage_ratio):
    """
    apply convolution operation to each layer's parameters

        :param large_state_dict: the global model's state dict
        :param conv_filter: the corresponding convolution parameters
        :param shrinkage_ratio: the shrinkage ratio of the model
        :return: convoluted state dict
    """

    result_state_dict = OrderedDict({})
    activation_function = Leaky_Decay_ReLU()

    for key, value in conv_filter.items():
        current_global_parameter = large_state_dict[key]

        if not current_global_parameter.requires_grad:
            if current_global_parameter.ndim != 0:
                result_state_dict[key] = current_global_parameter[
                    : int(shrinkage_ratio * len(current_global_parameter))
                ]
            # the num_batches_tracked of bn
            else:
                result_state_dict[key] = current_global_parameter
        elif "conv" in key or "linear" in key:
            result_state_dict[key] = activation_function(
                value(current_global_parameter)
            )
        else:
            result_state_dict[key] = None

    return result_state_dict


def deconvolution(small_state_dict, deconv_filter):
    """
    apply transposed convolution operation to each layer's parameters of each client

        :param small_state_dict: the small model's state dict
        :param deconv_filter: the corresponding transposed convolution parameters
        :return: deconvoluted state dict
    """

    result_state_dict = OrderedDict({})
    activation_function = Leaky_Decay_ReLU()
    sample_state_dict = utils.client_settings.get_model(0).state_dict()

    for key, value in deconv_filter.items():
        current_global_parameter = small_state_dict[key]

        if not current_global_parameter.requires_grad:
            if current_global_parameter.ndim != 0:
                if current_global_parameter.sum() == 0:
                    result_state_dict[key] = (
                        torch.zeros_like(sample_state_dict[key])
                        .requires_grad_(False)
                        .to(config.device)
                    )
                else:
                    result_state_dict[key] = (
                        torch.ones_like(sample_state_dict[key])
                        .requires_grad_(False)
                        .to(config.device)
                    )
            else:
                result_state_dict[key] = current_global_parameter
        elif "conv" in key or "linear" in key:
            result_state_dict[key] = activation_function(
                value(current_global_parameter)
            )
        else:
            result_state_dict[key] = None

    return result_state_dict


def weighted_parameter(
    aggregated_large_model, weight_vector, state_dict_list, example_number_list
):
    """
    calculate the weighted sum parameters given a list of model's state dict

        :param aggregated_large_model: sample model for aggregated large model
        :param weight_vector: weight vector for each client
        :param state_dict_list: a list of large model's state dict
        :param example_number_list: a list of example_number for each client
        :return: aggregated state dict
    """

    aggregated_state_dict = OrderedDict({})

    for key in aggregated_large_model.state_dict().keys():
        if weight_vector[key] is None:
            aggregated_state_dict[key] = sum(
                parameter * example_number
                for parameter, example_number in zip(
                    [state_dict[key] for state_dict in state_dict_list],
                    example_number_list,
                )
            ) / sum(example_number_list)
        else:
            aggregated_state_dict[key] = sum(
                weight * parameter * example_number
                for weight, parameter, example_number in zip(
                    weight_vector[key],
                    [state_dict[key] for state_dict in state_dict_list],
                    example_number_list,
                )
            ) / sum(example_number_list)

    return aggregated_state_dict


def train_conv_deconv_parameter(
    input_model,
    output_model,
    conv_deconv_filter,
    validation_loader,
    test_loader,
    learning_rate,
    shrinkage_ratio,
    server_round,
    conv_deconv=True,
    use_scheduler=True,
    model_size_index=None,
    client_id=None,
):
    """
    iteratively train the convolution or transposed convolution parameters

        :param input_model: global model in configure fit or reconstructed model in aggregate fit
        :param output_model: client model for fit or after fit
        :param conv_deconv_filter: the parameters of convolution or transposed convolution
        :param validation_loader: validation dataset
        :param test_loader: test dataset
        :param learning_rate: the current conv_deconv_learning_rate
        :param shrinkage_ratio: shrinkage ratio
        :param conv_deconv: True for applying convolution operations
        :param use_scheduler: True for use
        :return: a list of state dict, containing conv or deconv filter and accuracy
    """

    if server_round > 5:
        config.deconv_model_epoch = 10
        config.conv_model_epoch = 10

    criterion = nn.CrossEntropyLoss()
    state_dict_list, optimizer_list = [], []

    if model_size_index is not None:
        temp_file_name = "./state_dict/temp_{}.pth".format(model_size_index)
    else:
        temp_file_name = "./state_dict/temp{}.pth".format(client_id)

    for value in conv_deconv_filter.values():
        if value is not None:
            if isinstance(value, list):
                for element in value:
                    optimizer_list.append(
                        torch.optim.Adam(params=element.parameters(), lr=learning_rate)
                    )
            else:
                optimizer_list.append(
                    torch.optim.Adam(params=value.parameters(), lr=learning_rate)
                )
    scheduler_list = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=5,
            last_epoch=-1,
            verbose=False,
            eta_min=config.conv_deconv_eta_min,
        )
        for optimizer in optimizer_list
    ]

    for epoch in range(config.conv_model_epoch):
        # train
        for validation_data, validation_label in tqdm(validation_loader):
            validation_data = validation_data.to(config.device)
            validation_label = validation_label.to(config.device)

            result_state_dict = (
                convolution(
                    large_state_dict=input_model.state_dict(),
                    conv_filter=conv_deconv_filter,
                    shrinkage_ratio=shrinkage_ratio,
                )
                if conv_deconv
                else deconvolution(
                    small_state_dict=input_model.state_dict(),
                    deconv_filter=conv_deconv_filter,
                )
            )

            output_model.load_state_dict(result_state_dict, keep_graph=True)

            model_output = output_model(validation_data)
            loss = criterion(model_output, validation_label)

            for optimizer in optimizer_list:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in optimizer_list:
                optimizer.step()

        result_state_dict = (
            convolution(
                large_state_dict=input_model.state_dict(),
                conv_filter=conv_deconv_filter,
                shrinkage_ratio=shrinkage_ratio,
            )
            if conv_deconv
            else deconvolution(
                small_state_dict=input_model.state_dict(),
                deconv_filter=conv_deconv_filter,
            )
        )

        output_model.load_state_dict(result_state_dict, keep_graph=False)

        # test
        (
            validation_accuracy,
            validation_loss,
        ) = print_model_accuracy(
            output_model, validation_loader, epoch, "validation dataset"
        )
        test_accuracy, test_loss = print_model_accuracy(
            output_model, test_loader, epoch, "test dataset"
        )

        torch.save(conv_deconv_filter, temp_file_name)

        state_dict_list.append(
            {
                "epoch": epoch,
                "state_dict": output_model.state_dict(),
                "conv_deconv_filter": torch.load(temp_file_name),
                "validation_accuracy": validation_accuracy,
                "validation_loss": validation_loss,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
            }
        )

        if validation_accuracy < 40 or test_accuracy < 40:
            return state_dict_list

        if use_scheduler:
            for scheduler in scheduler_list:
                scheduler.step()

    return state_dict_list


def weighted_aggregate(
    weight_vector,
    weight_list: List[Tuple[NDArrays, int]],
    learning_rate: float,
    validation_loader,
    test_loader,
    use_scheduler=False,
):
    """
    get the aggregated model's state dict using weighted average aggregation

        :param weight_vector: the weight vector for each client and each layer
        :param weight_list: a list of parameter list after transposed convolution
        :param learning_rate: the learning rate for weight vector
        :param validation_loader: validation dataset
        :param test_loader: test dataset
        :param use_scheduler: True for use
        :return: a state dict list containing updated weight vector and accuracy
    """

    criterion = torch.nn.CrossEntropyLoss()
    output_state_dict_list, optimizer_list, state_dict_list = [], [], []
    aggregated_large_model = utils.client_settings.get_model(model_size_index=0)

    optimizer_list = []
    for value in weight_vector.values():
        if value is not None:
            optimizer_list.append(torch.optim.Adam(params=value, lr=learning_rate))

    scheduler_list = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            T_max=4,
            last_epoch=-1,
            verbose=False,
            optimizer=optimizer,
            eta_min=config.weight_vector_eta_min,
        )
        for optimizer in optimizer_list
    ]

    example_number_list = [element[1] for element in weight_list]
    weight_parameter_list = [element[0] for element in weight_list]

    for weight in weight_parameter_list:
        temp_model = utils.client_settings.get_model(model_size_index=0)
        utils.client_settings.set_parameters(temp_model, weight)
        state_dict_list.append(temp_model.state_dict())

    for epoch in range(config.weighted_aggregate_epoch):
        # train weight vector
        for validation_data, validation_label in tqdm(validation_loader):
            validation_data = validation_data.to(config.device)
            validation_label = validation_label.to(config.device)

            aggregated_state_dict = weighted_parameter(
                aggregated_large_model=aggregated_large_model,
                weight_vector=weight_vector,
                example_number_list=example_number_list,
                state_dict_list=state_dict_list,
            )
            aggregated_large_model.load_state_dict(
                aggregated_state_dict, keep_graph=True
            )

            model_output = aggregated_large_model(validation_data)
            loss = criterion(model_output, validation_label)

            for optimizer in optimizer_list:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in optimizer_list:
                optimizer.step()

        aggregated_state_dict = weighted_parameter(
            aggregated_large_model=aggregated_large_model,
            weight_vector=weight_vector,
            example_number_list=example_number_list,
            state_dict_list=state_dict_list,
        )
        aggregated_large_model.load_state_dict(aggregated_state_dict, keep_graph=False)

        (
            validation_accuracy,
            validation_loss,
        ) = print_model_accuracy(
            aggregated_large_model, validation_loader, epoch, "validation dataset"
        )
        test_accuracy, test_loss = print_model_accuracy(
            aggregated_large_model, test_loader, epoch, "test dataset"
        )

        torch.save(weight_vector, "./state_dict/temp.pth")

        output_state_dict_list.append(
            {
                "epoch": epoch,
                "state_dict": aggregated_large_model.state_dict(),
                "weight_vector": torch.load("./state_dict/temp.pth"),
                "validation_accuracy": validation_accuracy,
                "validation_loss": validation_loss,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
            }
        )

        if validation_accuracy < 15 or test_accuracy < 15:
            return state_dict_list

        if use_scheduler:
            for scheduler in scheduler_list:
                scheduler.step()

    return output_state_dict_list


def print_model_accuracy(model, data_loader, epoch, dataset_name):
    """
    get the model's accuracy and loss given a dataset

        :param model: model to be evaluated
        :param data_loader: data loader to be used
        :param epoch: epoch number during training
        :param dataset_name: a string of the dataset's name
        :return: total accuracy and average loss
    """

    criterion = nn.CrossEntropyLoss()
    correct, total, loss_list = 0, 0, []

    with torch.no_grad():
        for train_data, train_label in data_loader:
            train_data = train_data.to(config.device)
            train_label = train_label.to(config.device)

            model_output = model(train_data)
            _, predicted = torch.max(model_output, 1)
            loss_list.append(float(criterion(model_output, train_label)))

            correct += (predicted == train_label).sum().item()
            total += len(train_label)

        print(
            "epoch {}, model accuracy {}% on {}, average loss {}".format(
                epoch, correct / total * 100, dataset_name, np.average(loss_list)
            )
        )

    return correct / total * 100, np.average(loss_list)


def server_local_training(model, data_loader, server_round):
    """
    perform server-side local training

        :param model: the model to be trained
        :param data_loader: the data loader to be used
        :param server_round: current global communication round
        :return: the trained model
    """

    print("server-side local training")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    server_epoch = config.server_epoch if server_round == 1 else 10

    for epoch in range(server_epoch):
        # train
        for train_data, train_label in tqdm(data_loader):
            train_data = train_data.to(config.device)
            train_label = train_label.to(config.device)

            model_output = model(train_data)
            loss = criterion(model_output, train_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print_model_accuracy(model, data_loader, epoch, "validation dataset")

    return model


def convolution_process(model_size_index, validation_loader, test_loader, server_round):
    print("convolution model size {} now".format(model_size_index))
    large_model_path = "./state_dict/large_model.pth"
    conv_filter_path = "./state_dict/conv_filter{}.pth".format(
        model_size_index
    )

    large_model = utils.client_settings.get_model(0)
    large_model.load_state_dict(torch.load(large_model_path))
    small_model = utils.client_settings.get_model(model_size_index)
    current_conv_filter = torch.load(conv_filter_path)

    while True:
        state_dict_list = train_conv_deconv_parameter(
            input_model=large_model,
            output_model=small_model,
            conv_deconv=True,
            conv_deconv_filter=current_conv_filter,
            validation_loader=validation_loader,
            test_loader=test_loader,
            learning_rate=config.conv_deconv_learning_rate,
            shrinkage_ratio=math.pow(config.shrinkage_ratio, model_size_index),
            server_round=server_round,
            use_scheduler=False,
            model_size_index=model_size_index,
            client_id=None,
        )

        evaluate_list = [
            epoch_dict["test_accuracy"] - epoch_dict["test_loss"]
            for epoch_dict in state_dict_list
        ]
        dict_index = evaluate_list.index(max(evaluate_list))

        if (
            state_dict_list[dict_index]["test_accuracy"] < 40
            or state_dict_list[dict_index]["validation_accuracy"] < 40
        ):
            current_conv_filter = initialize_conv_deconv_parameter(
                model_size_index=model_size_index, conv_deconv=True
            )

            continue
        else:
            current_conv_filter = state_dict_list[dict_index]["conv_deconv_filter"]
            small_model.load_state_dict(
                state_dict_list[dict_index]["state_dict"], keep_graph=False
            )

            print(
                "Finished convolution of model size {}, selected index {}".format(
                    model_size_index, dict_index
                )
            )

            print_model_accuracy(
                model=small_model,
                data_loader=validation_loader,
                epoch=-1,
                dataset_name="validation dataset",
            )
            print_model_accuracy(
                model=small_model,
                data_loader=test_loader,
                epoch=-1,
                dataset_name="test dataset",
            )

            torch.save(current_conv_filter, conv_filter_path)

            break


def deconvolution_process(
    model_size_index,
    client_parameter,
    validation_loader,
    test_loader,
    client_id,
    server_round,
):
    large_model = utils.client_settings.get_model(model_size_index=0)
    client_model = utils.client_settings.get_model(model_size_index)
    pre_train_client(
        validation_loader=validation_loader,
        test_loader=test_loader,
        parameters=client_parameter,
        client_id=client_id,
        server_round=1,
    )
    client_model.load_state_dict(
        torch.load(
            "./state_dict/client_model{}.pth".format(client_id)
        )
    )
    current_deconv_filter = torch.load(
        "./state_dict/deconv_filter{}.pth".format(client_id)
    )

    while True:
        state_dict_list = train_conv_deconv_parameter(
            input_model=client_model,
            output_model=large_model,
            conv_deconv=False,
            conv_deconv_filter=current_deconv_filter,
            validation_loader=validation_loader,
            test_loader=test_loader,
            learning_rate=config.conv_deconv_learning_rate,
            server_round=server_round,
            shrinkage_ratio=None,
            use_scheduler=False,
            model_size_index=None,
            client_id=client_id,
        )

        evaluate_list = [
            epoch_dict["test_accuracy"] - epoch_dict["test_loss"]
            for epoch_dict in state_dict_list
        ]
        dict_index = evaluate_list.index(max(evaluate_list))

        if (
            state_dict_list[dict_index]["test_accuracy"] < 80
            or state_dict_list[dict_index]["validation_accuracy"] < 80
        ):
            current_deconv_filter = initialize_conv_deconv_parameter(
                model_size_index=model_size_index, conv_deconv=False
            )
            torch.save(
                current_deconv_filter,
                "./state_dict/deconv_filter{}.pth".format(client_id),
            )
            continue
        else:
            current_deconv_filter = state_dict_list[dict_index]["conv_deconv_filter"]
            large_model.load_state_dict(
                state_dict_list[dict_index]["state_dict"], keep_graph=False
            )

            print(
                "Finished transposed convolution of client {}, selected index {}".format(
                    client_id, dict_index
                )
            )

            print_model_accuracy(
                model=large_model,
                data_loader=validation_loader,
                epoch=-1,
                dataset_name="validation dataset",
            )
            print_model_accuracy(
                model=large_model,
                data_loader=test_loader,
                epoch=-1,
                dataset_name="test dataset",
            )

            torch.save(
                current_deconv_filter,
                "./state_dict/deconv_filter{}.pth".format(client_id),
            )

            torch.save(
                large_model.state_dict(),
                "./state_dict/client_model{}.pth".format(client_id),
            )

            break


def pre_train_client(
    validation_loader, test_loader, parameters, client_id, server_round
):
    model_size_index = config.shrinkage_ratio_exp[client_id]

    client_model = utils.client_settings.get_model(model_size_index)
    utils.client_settings.set_parameters(
        client_model, parameters_to_ndarrays(parameters)
    )

    print("client {} pre-training".format(client_id))

    server_local_training(
        model=client_model, data_loader=validation_loader, server_round=server_round
    )

    print_model_accuracy(client_model, validation_loader, -1, "validation dataset")
    print_model_accuracy(client_model, test_loader, -1, "test dataset")

    torch.save(
        client_model.state_dict(),
        "./state_dict/client_model{}.pth".format(client_id),
    )
