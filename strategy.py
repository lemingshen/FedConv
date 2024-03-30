import math
import flwr
import torch
import config
import utils.client_settings
import utils.data_processing
import utils.federated_settings
import torch.multiprocessing as mp

from logging import WARNING
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import Callable, Union, Optional, List, Tuple, Dict
from flwr.common import EvaluateRes, EvaluateIns, FitIns, FitRes
from flwr.common import MetricsAggregationFn, NDArrays, Parameters
from flwr.common import (
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    GetPropertiesIns,
)

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
mp.set_start_method("spawn", force=True)


# a self designed federated strategy - FedConv
class FederatedCustom(flwr.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.conv_deconv_learning_rate = config.conv_deconv_learning_rate
        self.weight_vector_learning_rate = config.weight_vector_learning_rate
        self.conv_deconv_eta_min = config.conv_deconv_eta_min

        # {cid: client_id}
        self.client_dict = {}
        self.client_conv_parameter = []
        self.client_deconv_parameter = []
        self.client_loss_list = [0] * config.client_number
        self.weight_vector = None

    def __repr__(self) -> str:
        response = f"HeteroFL(accept_failures={self.accept_failures})"

        return response

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """return the sample size and the required number of available clients"""
        client_number = int(num_available_clients * self.fraction_fit)

        return max(client_number, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """use a fraction of available clients for evaluation"""
        client_number = int(num_available_clients * self.fraction_evaluate)

        return max(client_number, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        initialize global model parameters

        :param client_manager:
        :return:
        """

        print("===================== Initializing parameters =====================")
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory

        # initialize conv and deconv parameters
        for model_size_index in range(config.model_size_number):
            self.client_conv_parameter.append(
                utils.federated_settings.initialize_conv_deconv_parameter(
                    model_size_index=model_size_index, conv_deconv=True
                )
            )

        for client_index in range(config.client_number):
            model_size_index = config.shrinkage_ratio_exp[client_index]

            self.client_deconv_parameter.append(
                utils.federated_settings.initialize_conv_deconv_parameter(
                    model_size_index=model_size_index, conv_deconv=False
                )
            )

        # initialize weight vector
        self.weight_vector = utils.federated_settings.initialize_weight_vector()

        print("===================== Finished initializing =====================")

        # map cid and client id
        sample_size, _ = self.num_fit_clients(client_manager.num_available())

        client_list = client_manager.sample(sample_size, None)
        instruction = GetPropertiesIns(config={})

        for current_client in client_list:
            response = current_client.get_properties(instruction, timeout=None)
            properties = response.properties
            current_client_id = properties["client_id"]

            self.client_dict[current_client.cid] = current_client_id

        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            # no evaluation function provided
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        evaluation_result = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if evaluation_result is None:
            return None

        loss, metrics = evaluation_result

        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round for client local training
        In FedConv, client models with different sizes are generated via convolutional compression

            :param server_round: current global communication round
            :param parameters: parameters of the global model
            :param client_manager:
            :return: a list of tuple with client_proxy and fit instruction
        """

        # modify the Conv & TC learning rate every 10 server round
        if server_round % config.learning_rate_change_step == 0:
            if self.conv_deconv_learning_rate > 0.0001:
                self.conv_deconv_learning_rate /= 2
                self.conv_deconv_eta_min /= 2
            if self.weight_vector_learning_rate > 0.0001:
                self.weight_vector_learning_rate /= 2

        # first calculate convoluted parameter
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
        large_model = utils.client_settings.get_model(model_size_index=0)
        utils.client_settings.set_parameters(
            large_model, parameters_to_ndarrays(parameters)
        )

        # %%
        # first perform server side local training
        utils.federated_settings.server_local_training(
            model=large_model, data_loader=validation_loader, server_round=server_round
        )
        torch.save(large_model.state_dict(), "./state_dict/large_model.pth")

        for model_size_index in range(1, config.model_size_number):
            torch.save(
                self.client_conv_parameter[model_size_index],
                "./state_dict/conv_filter{}.pth".format(
                    model_size_index
                ),
            )

        convolution_process_list = []
        for model_size_index in range(1, config.model_size_number):
            process = mp.Process(
                target=utils.federated_settings.convolution_process,
                args=(model_size_index, validation_loader, test_loader, server_round),
            )
            convolution_process_list.append(process)

        # use multiple processes to speed up the convolutional compression
        for process in convolution_process_list:
            process.start()

        for process in convolution_process_list:
            process.join()

        for model_size_index in range(1, config.model_size_number):
            self.client_conv_parameter[model_size_index] = torch.load(
                "./state_dict/conv_filter{}.pth".format(model_size_index)
            )

        # %% setting up client fit configuration
        configuration = {}
        client_config_pairs = []

        if self.on_fit_config_fn is not None:
            # custom fit config function provided
            configuration = self.on_fit_config_fn(server_round)

        # sample clients
        sample_size, min_client_number = self.num_fit_clients(
            client_manager.num_available()
        )
        client_list = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_client_number
        )

        # calculate the shrunken parameter according to the conv_parameter list and transmit to the client
        for current_client in client_list:
            current_client_id = self.client_dict[current_client.cid]
            model_size_index = config.shrinkage_ratio_exp[current_client_id]

            if model_size_index == 0:
                current_fit_instruction = FitIns(
                    ndarrays_to_parameters(
                        utils.client_settings.get_parameters(large_model)
                    ),
                    configuration,
                )
            else:
                current_conv_parameter = self.client_conv_parameter[model_size_index]
                convoluted_state_dict = utils.federated_settings.convolution(
                    large_state_dict=large_model.state_dict(),
                    conv_filter=current_conv_parameter,
                    shrinkage_ratio=math.pow(config.shrinkage_ratio, model_size_index),
                )

                small_model = utils.client_settings.get_model(model_size_index)
                small_model.load_state_dict(convoluted_state_dict, keep_graph=False)

                convoluted_parameters = utils.client_settings.get_parameters(
                    small_model
                )
                convoluted_parameters = ndarrays_to_parameters(convoluted_parameters)
                current_fit_instruction = FitIns(convoluted_parameters, configuration)

            client_config_pairs.append((current_client, current_fit_instruction))

        # return client/config pairs
        return client_config_pairs

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        configure the current round for client-side evaluation

        :param server_round: current global communication round
        :param parameters: parameters of the global model
        :param client_manager: a list of client
        :return: a list of tuple with client_proxy and evaluate instruction
        """

        if self.fraction_evaluate == 0.0:
            return []

        # parameters and config
        configuration, client_config_pairs = {}, []

        if self.on_evaluate_config_fn is not None:
            # custom evaluation config function provided
            configuration = self.on_evaluate_config_fn(server_round)

        # sample clients
        sample_size, min_client_number = self.num_evaluation_clients(
            client_manager.num_available()
        )
        client_list = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_client_number
        )

        large_model = utils.client_settings.get_model(model_size_index=0)
        utils.client_settings.set_parameters(
            large_model, parameters_to_ndarrays(parameters)
        )

        for current_client in client_list:
            current_client_id = self.client_dict[current_client.cid]
            model_size_index = config.shrinkage_ratio_exp[current_client_id]

            if model_size_index == 0:
                current_evaluate_instruction = EvaluateIns(
                    ndarrays_to_parameters(
                        utils.client_settings.get_parameters(large_model)
                    ),
                    configuration,
                )
            else:
                current_conv_filter = self.client_conv_parameter[model_size_index]
                convoluted_state_dict = utils.federated_settings.convolution(
                    large_state_dict=large_model.state_dict(),
                    conv_filter=current_conv_filter,
                    shrinkage_ratio=math.pow(config.shrinkage_ratio, model_size_index),
                )

                small_model = utils.client_settings.get_model(
                    model_size_index=model_size_index
                )
                small_model.load_state_dict(convoluted_state_dict, keep_graph=False)

                current_evaluate_instruction = EvaluateIns(
                    ndarrays_to_parameters(
                        utils.client_settings.get_parameters(small_model)
                    ),
                    configuration,
                )

            client_config_pairs.append((current_client, current_evaluate_instruction))

            # return client/config pairs
        return client_config_pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        first apply transposed convolution to all the clients' parameters

        then aggregate fit results using weighted average

        :param server_round: current global communication round
        :param results: a list of client_proxy and their fit response, containing parameters
        :param failures:
        :return: the aggregated parameters and metrics
        """
        print(
            "=============== FIT aggregating, server round {} ===============".format(
                server_round
            )
        )

        if not results:
            return None, {}

        # do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # load data loader and some other settings
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

        # %%
        # apply TC on each client's parameter before aggregation
        (
            weight_list,
            client_id_list,
            client_parameter_list,
            client_example_number_list,
        ) = ([], [], [], [])

        (
            temp_client_id_list,
            temp_client_parameter_list,
            temp_client_example_number_list,
        ) = ([], [], [])

        for client_proxy, fit_response in results:
            current_cid = client_proxy.cid
            current_client_id = self.client_dict[current_cid]

            if config.shrinkage_ratio_exp[current_client_id] == 0:
                temp_client_id_list.append(current_client_id)
                temp_client_parameter_list.append(fit_response.parameters)
                temp_client_example_number_list.append(fit_response.num_examples)
            else:
                client_id_list.append(current_client_id)
                client_parameter_list.append(fit_response.parameters)
                client_example_number_list.append(fit_response.num_examples)
                torch.save(
                    self.client_deconv_parameter[current_client_id],
                    "./state_dict/deconv_filter{}.pth".format(
                        current_client_id
                    ),
                )

        pre_train_list = []
        for current_client_id, current_parameter, current_example_number in zip(
            temp_client_id_list,
            temp_client_parameter_list,
            temp_client_example_number_list,
        ):
            process = mp.Process(
                target=utils.federated_settings.pre_train_client,
                args=(
                    validation_loader,
                    test_loader,
                    current_parameter,
                    current_client_id,
                    server_round,
                ),
            )
            pre_train_list.append(process)

        # use multiple processes to speed up the TC dilation process
        for process in pre_train_list:
            process.start()

        for process in pre_train_list:
            process.join()

        for current_client_id, current_parameter, current_example_number in zip(
            temp_client_id_list,
            temp_client_parameter_list,
            temp_client_example_number_list,
        ):
            temp_model = utils.client_settings.get_model(0)
            temp_model.load_state_dict(
                torch.load(
                    "./state_dict/client_model{}.pth".format(current_client_id)
                )
            )

            weight_list.append(
                (
                    utils.client_settings.get_parameters(temp_model),
                    current_example_number,
                )
            )

        deconvolution_process_list = []
        for current_client_id, current_parameter, current_example_number in zip(
            client_id_list, client_parameter_list, client_example_number_list
        ):
            print(
                "Processing transposed convolution of client {}".format(
                    current_client_id
                )
            )
            model_size_index = config.shrinkage_ratio_exp[current_client_id]
            process = mp.Process(
                target=utils.federated_settings.deconvolution_process,
                args=(
                    model_size_index,
                    current_parameter,
                    validation_loader,
                    test_loader,
                    current_client_id,
                    server_round,
                ),
            )
            deconvolution_process_list.append(
                (process, model_size_index, current_client_id, current_example_number)
            )

            if len(deconvolution_process_list) == 2:
                for process in deconvolution_process_list:
                    process[0].start()

                for process in deconvolution_process_list:
                    process[0].join()

                # deal with output
                for process in deconvolution_process_list:
                    model_size_index = process[1]
                    temp_client_id = process[2]
                    temp_example_number = process[3]

                    temp_client_model = utils.client_settings.get_model(0)
                    temp_client_model.load_state_dict(
                        torch.load(
                            "./state_dict/client_model{}.pth".format(temp_client_id)
                        )
                    )

                    weight_list.append(
                        (
                            utils.client_settings.get_parameters(temp_client_model),
                            temp_example_number,
                        )
                    )

                    self.client_deconv_parameter[current_client_id] = torch.load(
                        "./state_dict/deconv_filter{}.pth".format(current_client_id)
                    )

                deconvolution_process_list.clear()

        # %%
        # weighted average aggregation
        large_model = utils.client_settings.get_model(model_size_index=0)

        while True:
            current_weight_vector = self.weight_vector

            state_dict_list = utils.federated_settings.weighted_aggregate(
                weight_list=weight_list,
                weight_vector=current_weight_vector,
                learning_rate=self.weight_vector_learning_rate,
                validation_loader=validation_loader,
                test_loader=test_loader,
                use_scheduler=False,
            )

            evaluate_list = [
                epoch_dict["test_accuracy"] - epoch_dict["test_loss"]
                for epoch_dict in state_dict_list
            ]
            dict_index = evaluate_list.index(max(evaluate_list))

            if (
                state_dict_list[dict_index]["test_accuracy"] < 50
                or state_dict_list[dict_index]["validation_accuracy"] < 50
            ):
                print(
                    "Something went wrong in current aggregation iteration, re-aggregating . . ."
                )
                self.weight_vector = utils.federated_settings.initialize_weight_vector()

                continue
            else:
                current_weight_vector = state_dict_list[dict_index]["weight_vector"]
                large_state_dict = state_dict_list[dict_index]["state_dict"]
                large_model.load_state_dict(large_state_dict, keep_graph=False)

                test_accuracy, _ = utils.federated_settings.print_model_accuracy(
                    model=large_model,
                    data_loader=test_loader,
                    epoch=-1,
                    dataset_name="test dataset",
                )

                print(
                    "Finished weighted average aggregation, selected index {}".format(
                        dict_index
                    )
                )

                utils.federated_settings.print_model_accuracy(
                    model=large_model,
                    data_loader=validation_loader,
                    epoch=-1,
                    dataset_name="validation dataset",
                )
                utils.federated_settings.print_model_accuracy(
                    model=large_model,
                    data_loader=test_loader,
                    epoch=-1,
                    dataset_name="test dataset",
                )

                self.weight_vector = current_weight_vector
                torch.save(
                    self.weight_vector,
                    "{}/{}/server_round_{}_weight_vector.pth".format(
                        config.weight_vector_save_path,
                        config.dataset_dict[config.dataset_type][config.dataset_index],
                        server_round,
                    ),
                )

                break

        aggregated_parameters = utils.client_settings.get_parameters(large_model)
        aggregated_parameters = ndarrays_to_parameters(aggregated_parameters)

        # aggregate custom metrics if aggregation function was provided
        aggregated_metrics = {}

        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (response.num_examples, response.metrics) for _, response in results
            ]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # long log this warning once
            log(WARNING, "Not fit_metrics_aggregation_fn provides")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        aggregate evaluation losses using weighted average

        :param server_round: the current global communication round
        :param results: a list of client proxy with evaluate response from clients
        :param failures:
        :return: aggregated loss and metrics
        """

        print(
            f"=============== server round: {server_round} EVALUATION aggregating ==============="
        )

        if not results:
            return None, {}

        # do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # aggregate loss
        aggregated_loss = utils.federated_settings.weighted_loss_avg(
            [
                (evaluate_response.num_examples, evaluate_response.loss)
                for _, evaluate_response in results
            ]
        )

        # aggregate custom metrics if aggregation function was provided
        aggregated_metrics = {}

        if self.evaluate_metrics_aggregation_fn:
            evaluate_metrics = [
                (response.num_examples, response.metrics) for _, response in results
            ]
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(evaluate_metrics)
        elif server_round == 1:  # only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return aggregated_loss, aggregated_metrics
