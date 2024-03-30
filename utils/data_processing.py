import torch
import config
import numpy as np

from torch.utils.data import DataLoader, TensorDataset


def load_dataset(
        client_id: int = None,
        client_or_server: bool = True,
        train_dataset: bool = False,
        validation_dataset: bool = False,
        test_dataset: bool = False
):
    """
    load dataset for client or server

    dataset order: train, validation, test, example_number

        :param client_id: the client id
        :param client_or_server: True for client, False for server
        :param train_dataset: True for need train dataset
        :param validation_dataset: True for need validation dataset
        :param test_dataset: True for need test dataset
        :return: train_loader, validation_loader, test_loader, example_number
    """

    train_loader, validation_loader, test_loader = None, None, None

    # load client-side dataset
    if client_or_server:
        print('===================== Loading client datasets =====================')

        client_train_data = np.load('{}/{}/{}/alpha_{}/train_data_client_{}.npy'.format(
            config.dataset_root_path,
            config.dataset_type,
            config.dataset_dict[config.dataset_type][config.dataset_index],
            config.alpha,
            client_id
        ), allow_pickle=True)
        client_test_data = np.load('{}/{}/{}/alpha_{}/test_data_client_{}.npy'.format(
            config.dataset_root_path,
            config.dataset_type,
            config.dataset_dict[config.dataset_type][config.dataset_index],
            config.alpha,
            client_id
        ), allow_pickle=True)

        client_train_label = np.load('{}/{}/{}/alpha_{}/train_label_client_{}.npy'.format(
            config.dataset_root_path,
            config.dataset_type,
            config.dataset_dict[config.dataset_type][config.dataset_index],
            config.alpha,
            client_id
        ), allow_pickle=True)
        client_test_label = np.load('{}/{}/{}/alpha_{}/test_label_client_{}.npy'.format(
            config.dataset_root_path,
            config.dataset_type,
            config.dataset_dict[config.dataset_type][config.dataset_index],
            config.alpha,
            client_id
        ), allow_pickle=True)

        print('Loading client data succeed, client id: {}'.format(client_id))
        print('client train data size: {}'.format(client_train_data.shape))
        print('client train label size: {}'.format(client_train_label.shape))
        print('client test data size: {}'.format(client_test_data.shape))
        print('client test label size: {}'.format(client_test_label.shape))

        # convert numpy to tensor
        client_train_data = torch.from_numpy(client_train_data).float()
        client_train_label = torch.from_numpy(client_train_label).long()
        client_test_data = torch.from_numpy(client_test_data).float()
        client_test_label = torch.from_numpy(client_test_label).long()

        # generate data loader
        client_train_loader = DataLoader(
            dataset=TensorDataset(client_train_data, client_train_label),
            shuffle=True, batch_size=config.batch_size
        )
        client_test_loader = DataLoader(
            dataset=TensorDataset(client_test_data, client_test_label),
            shuffle=True, batch_size=config.batch_size
        )

        example_number = {
            'train_set': len(client_train_label),
            'test_set': len(client_test_label)
        }

        return client_train_loader, client_test_loader, example_number
    # load server-side dataset
    # the order is train, validation, test, example_number
    else:
        if train_dataset:
            print('===================== Loading server train datasets =====================')

            train_data = np.load('{}/{}/{}/train_data.npy'.format(
                config.dataset_root_path,
                config.dataset_type,
                config.dataset_dict[config.dataset_type][config.dataset_index],
            ), allow_pickle=True)
            train_label = np.load('{}/{}/{}/train_label.npy'.format(
                config.dataset_root_path,
                config.dataset_type,
                config.dataset_dict[config.dataset_type][config.dataset_index],
            ), allow_pickle=True)

            print('server train data size: {}'.format(train_data.shape))
            print('server train label size: {}'.format(train_label.shape))

            train_data = torch.from_numpy(train_data).float()
            train_label = torch.from_numpy(train_label).long()

            train_loader = DataLoader(
                dataset=TensorDataset(train_data, train_label),
                shuffle=True, batch_size=config.batch_size
            )

        if validation_dataset:
            print('===================== Loading server validation datasets =====================')

            validation_data = np.load('{}/{}/{}/validation_data.npy'.format(
                config.dataset_root_path,
                config.dataset_type,
                config.dataset_dict[config.dataset_type][config.dataset_index],
            ), allow_pickle=True)
            validation_label = np.load('{}/{}/{}/validation_label.npy'.format(
                config.dataset_root_path,
                config.dataset_type,
                config.dataset_dict[config.dataset_type][config.dataset_index],
            ), allow_pickle=True)

            print('server validation data size: {}'.format(validation_data.shape))
            print('server validation label size: {}'.format(validation_label.shape))

            validation_data = torch.from_numpy(validation_data).float()
            validation_label = torch.from_numpy(validation_label).long()

            validation_loader = DataLoader(
                dataset=TensorDataset(validation_data, validation_label),
                shuffle=True, batch_size=config.batch_size
            )

        if test_dataset:
            test_data = np.load('{}/{}/{}/test_data.npy'.format(
                config.dataset_root_path,
                config.dataset_type,
                config.dataset_dict[config.dataset_type][config.dataset_index],
            ), allow_pickle=True)
            test_label = np.load('{}/{}/{}/test_label.npy'.format(
                config.dataset_root_path,
                config.dataset_type,
                config.dataset_dict[config.dataset_type][config.dataset_index],
            ), allow_pickle=True)

            print('server test data size: {}'.format(test_data.shape))
            print('server test label size: {}'.format(test_label.shape))

            test_data = torch.from_numpy(test_data).float()
            test_label = torch.from_numpy(test_label).long()

            test_loader = DataLoader(
                dataset=TensorDataset(test_data, test_label),
                shuffle=True, batch_size=config.batch_size
            )

        example_number = {
            'train_set': len(train_label) if train_dataset else None,
            'validation_set': len(validation_label) if validation_dataset else None,
            'test_set': len(test_label) if test_dataset else None
        }

        return (
            train_loader if train_dataset else None,
            validation_loader if validation_dataset else None,
            test_loader if test_dataset else None,
            example_number
        )


def load_client_sample_number(client_id):
    return len(np.load('{}/{}/{}/alpha_{}/train_label_client_{}.npy'.format(
        config.dataset_root_path,
        config.dataset_type,
        config.dataset_dict[config.dataset_type][config.dataset_index],
        config.alpha,
        client_id
    ), allow_pickle=True))
