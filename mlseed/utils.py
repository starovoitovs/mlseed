import yaml
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config
