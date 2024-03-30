import config
import strategy
import flwr as fl
import flwr.server.strategy
import utils.client_settings
import utils.data_processing
import utils.federated_settings
import torch.multiprocessing as mp


def main():
    model = utils.client_settings.get_model(model_size_index=0)
    parameters = utils.client_settings.get_parameters(model)

    FedAvg = flwr.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.client_number,
        min_evaluate_clients=config.client_number,
        min_available_clients=config.client_number,
        initial_parameters=fl.common.ndarrays_to_parameters(parameters),
        on_fit_config_fn=utils.federated_settings.fit_config,
        on_evaluate_config_fn=utils.federated_settings.evaluate_config,
        evaluate_fn=utils.federated_settings.federated_evaluation,
    )

    # use our customized FL strategy
    customized_strategy = strategy.FederatedCustom(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.client_number,
        min_evaluate_clients=config.client_number,
        min_available_clients=config.client_number,
        initial_parameters=fl.common.ndarrays_to_parameters(parameters),
        on_fit_config_fn=utils.federated_settings.fit_config,
        on_evaluate_config_fn=utils.federated_settings.evaluate_config,
        evaluate_fn=utils.federated_settings.federated_evaluation,
    )

    fl.server.start_server(
        server_address='0.0.0.0:' + config.server_port,
        config=fl.server.ServerConfig(
            num_rounds=config.communication_round,
            round_timeout=3600000
        ),
        strategy=customized_strategy,
        # strategy=FedAvg,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    main()
