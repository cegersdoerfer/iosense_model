import os
import json

IOSENSE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def load_model_config():
    with open(os.path.join(IOSENSE_ROOT,'iosense_model', 'config_files', 'model_config.json'), 'r') as f:
        return json.load(f)

def load_train_config():
    with open(os.path.join(IOSENSE_ROOT,'iosense_model', 'config_files', 'train_config.json'), 'r') as f:
        return json.load(f)
    
def load_data_config():
    with open(os.path.join(IOSENSE_ROOT,'iosense_model', 'config_files', 'data_config.json'), 'r') as f:
        return json.load(f)

def load_cluster_config():
    with open(os.path.join(IOSENSE_ROOT, 'client', 'cluster_config.json'), 'r') as f:
        config = json.load(f)
    print("Cluster configuration loaded.")
    return config