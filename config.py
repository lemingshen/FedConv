import torch

# global config
evaluation_root_path = "./Results"
dataset_root_path = './Datasets'

# server and client config
server_address = "localhost"
server_port = "8080"

# data set related config
# the type of model, 0 for MNIST CNN, 1 for CIFAR10 ResNet18W, 2 for CIFAR100 Wide_ResNet, 3 for WIFI CSI CNN
dataset_dict = {
    "Image": ["MNIST", "CIFAR10", "CINIC10"],
    "HAR": ["WiAR", "Depth_Camera", "HARBox"],
}
dataset_type = "Image"
dataset_index = 0

# training related config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_size = 0.2
batch_size = 32
client_learning_rate = 0.001
conv_deconv_learning_rate = 0.001
conv_deconv_eta_min = 0.0001
weight_vector_learning_rate = 0.001
weight_vector_eta_min = 0.0001
learning_rate_change_step = 25

# conv and deconv config
conv_model_epoch = 20
deconv_model_epoch = 20
server_epoch = 10
weighted_aggregate_epoch = 20
negative_slope = 0.001
positive_slope = 0.85

# testing related config
accuracy_threshold = 0.7
weight_vector_save_path = '{}/weight_vector'.format(evaluation_root_path)
evaluation_save_path = '{}/evaluation'.format(evaluation_root_path)
memory_footprint_save_path = '{}/memory'.format(evaluation_root_path)
server_evaluation_dict = {}
class_number = {
    "Image": [10, 10, 10],
    "HAR": [12, 5, 5]
}

# federated learning related hyperparameter
client_number = 10
local_epoch = 5
communication_round = 100
alpha = 10000
use_FedBN = False

# shrinkage ratio configuration
shrinkage_ratio = 0.75
model_size_number = 3
shrinkage_ratio_exp = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
