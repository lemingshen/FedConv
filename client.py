import os
import json
import time
import config
import argparse
import threading
import flwr as fl
import utils.client_settings
import utils.data_processing

from typing import Dict
from pympler import asizeof
from torch.utils.data import DataLoader
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Code
from flwr.common import GetPropertiesIns, GetPropertiesRes, EvaluateIns, FitIns
from flwr.common import EvaluateRes, FitRes, GetParametersIns, GetParametersRes, Status


class FlowerClient(fl.client.Client):
    def __init__(
            self,
            model,
            client_id: int,
            local_epoch: int,
            train_loader: DataLoader,
            test_loader: DataLoader,
            example_number: Dict[str, int]
    ) -> None:
        self.pid = os.getpid()
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.example_number = example_number
        self.local_epoch = local_epoch
        self.fit_dict = {}
        self.evaluation_dict = {}

        self.fit_file_path = '{}/{}/client_{}_fit.json'.format(
            config.evaluation_save_path,
            config.dataset_dict[config.dataset_type][config.dataset_index],
            self.client_id
        )
        self.evaluation_file_path = '{}/{}/client_{}_evaluate.json'.format(
            config.evaluation_save_path,
            config.dataset_dict[config.dataset_type][config.dataset_index],
            self.client_id
        )

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        properties = {
            'client_id': self.client_id
        }

        return GetPropertiesRes(
            status=Status(
                code=Code.OK,
                message='success'
            ),
            properties=properties
        )

    def get_parameters(self, instruction: GetParametersIns) -> GetParametersRes:
        print(f'[Client {self.client_id}] get parameters')

        # get parameters as a list of numpy ndarray
        array = utils.client_settings.get_parameters(self.model)

        # serialize ndarray into a parameters object
        parameters = ndarrays_to_parameters(array)

        # build and return response
        status = Status(
            code=Code.OK,
            message="success"
        )

        return GetParametersRes(
            status=status,
            parameters=parameters
        )

    def fit(self, instruction: FitIns) -> FitRes:
        """
        the fit process for the client, client only perform local training

        :param instruction: the fit instruction for this client
        :return: fit response
        """

        if self.client_id in [3, 4, 5]:
            time.sleep(20)
        elif self.client_id in [6, 7, 8, 9]:
            time.sleep(40)

        server_round = instruction.config['server_round']

        print("=============== [Client {}] FIT, current server round: {} ===============".format(
            self.client_id, server_round
        ))

        start_time = time.time()

        # deserialize parameters to numpy ndarray
        parameter_original = instruction.parameters
        array_original = parameters_to_ndarrays(parameter_original)

        # update local model, train, get updated parameters
        utils.client_settings.set_parameters(self.model, array_original)

        # %% training model process
        # train the local model
        loss_list, accuracy_list, time_list = utils.client_settings.train_model(
            model=self.model,
            client_id=self.client_id,
            server_round=server_round,
            train_loader=self.train_loader,
            epoch_number=self.local_epoch,
            example_number=self.example_number
        )

        end_time = time.time()

        # save the loss list and accuracy list into the dict
        current_round_dict = {
            "training_loss": loss_list,
            "training_accuracy": accuracy_list,
            "average_time": time_list,
            'wall_clock_time': end_time - start_time,
            'training_network': asizeof.asizeof(instruction) / 1024 / 1024  # in MB
        }
        self.fit_dict[server_round] = current_round_dict

        with open(self.fit_file_path, 'w') as fit_file:
            json.dump(self.fit_dict, fit_file, indent=4)

        # %% testing model process
        # use local test dataset to evaluate the model for personalization evaluation
        loss, accuracy = utils.client_settings.test_model(
            client_id=self.client_id,
            server_round=server_round,
            model=self.model,
            test_loader=self.test_loader,
            example_number=self.example_number
        )

        # save loss list and accuracy list into the dict
        current_round_dict = {
            'testing_loss': loss,
            'testing_accuracy': accuracy,
        }

        self.evaluation_dict[server_round] = current_round_dict

        with open(self.evaluation_file_path, 'w') as evaluation_file:
            json.dump(self.evaluation_dict, evaluation_file, indent=4)

        # save fit metric into fit_file
        array_updated = utils.client_settings.get_parameters(self.model)

        # serialize ndarray into a parameter object
        parameter_updated = ndarrays_to_parameters(array_updated)

        # build and return response
        status = Status(
            code=Code.OK,
            message="success"
        )

        return FitRes(
            status=status,
            parameters=parameter_updated,
            num_examples=self.example_number["train_set"],
            metrics={
                "accuracy": accuracy_list[-1],
                "loss": loss_list[-1],
                "client_id": self.client_id
            }
        )

    def evaluate(self, instruction: EvaluateIns) -> EvaluateRes:
        """
        evaluate the client's model

        :param instruction: evaluate instruction for the current client
        :return: evaluate response
        """

        server_round = instruction.config['server_round']

        print("=============== [Client {}] EVALUATE, current server round: {} ===============".format(
            self.client_id, server_round
        ))

        # deserialize parameters to numpy ndarray
        # parameter_original = instruction.parameters
        # array_original = parameters_to_ndarrays(parameter_original)
        #
        # utils.client_settings.set_parameters(self.model, array_original)
        # loss, accuracy = utils.client_settings.test_model(
        #     model=self.model,
        #     client_id=self.client_id,
        #     server_round=server_round,
        #     test_loader=self.test_loader,
        #     example_number=self.example_number
        # )

        # build and return response
        status = Status(
            code=Code.OK,
            message="success"
        )

        return EvaluateRes(
            status=status,
            loss=0.1,
            num_examples=10,
            metrics={"accuracy": 10}
        )


def parse_args():
    parse = argparse.ArgumentParser(description='create a Flower client')

    parse.add_argument('--client_id', default=0, type=int, help='the id of the created client')

    arguments = parse.parse_args()

    return arguments


def main(client_id, train_loader, test_loader, example_number):
    memory_dict = {}
    model_size_index = config.shrinkage_ratio_exp[client_id]
    model = utils.client_settings.get_model(model_size_index=model_size_index)

    # monitor the memory usage every 1 second
    # when in test mode, delete the thread to save memory and cpu
    timer_thread = threading.Thread(
        target=utils.client_settings.timer,
        args=(
            os.getpid(),
            memory_dict,
            client_id
        )
    )
    timer_thread.start()

    client = FlowerClient(
        model=model,
        client_id=client_id,
        train_loader=train_loader,
        test_loader=test_loader,
        example_number=example_number,
        local_epoch=config.local_epoch
    )

    fl.client.start_client(
        server_address='{}:{}'.format(
            config.server_address,
            config.server_port
        ),
        client=client,
    )


if __name__ == '__main__':
    args = parse_args()

    current_client_id = args.client_id

    client_train_loader, client_test_loader, client_example_number = utils.data_processing.load_dataset(
        client_id=current_client_id, client_or_server=True,
        train_dataset=False, validation_dataset=False, test_dataset=False
    )

    main(
        client_id=current_client_id,
        train_loader=client_train_loader,
        test_loader=client_test_loader,
        example_number=client_example_number
    )
