import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_model_config, load_data_config, load_train_config, IOSENSE_ROOT
from model_training.loader import MetricsDataset








class SensitivityModel(nn.Module):
    def __init__(self, devices, features, hidden_size=16, server_out_size = 8, output_size=1, server_emb_size=32):
        self.devices = devices
        self.features = features
        super(SensitivityModel, self).__init__()
        mdt_input_width = len(features['stats']) + len(features['mdt_trace']) + 1
        print('mdt_input_width: ', mdt_input_width)
        ost_input_width = len(features['stats']) + len(features['ost_trace']) + 1
        print('ost_input_width: ', ost_input_width)
        self.server_out_size = server_out_size
        self.mdt_fc = nn.Linear(mdt_input_width, server_emb_size)
        self.mdt_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.mdt_fc_out = nn.Linear(hidden_size, server_out_size)
        
        self.ost_fc = nn.Linear(ost_input_width, server_emb_size)
        self.ost_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.ost_fc_out = nn.Linear(hidden_size, server_out_size)
        self.fc_bridge = nn.Linear(len(devices['mdt'])*server_out_size + len(devices['ost'])*server_out_size, hidden_size)
        #self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        if output_size == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.Softmax(dim=1)

    def mdt_forward(self, mdt):
        mdt = mdt.view(-1, mdt.shape[-1])
        mdt = self.mdt_fc(mdt)
        mdt = self.relu(mdt)
        mdt = self.mdt_fc_hidden(mdt)
        mdt = self.relu(mdt)
        mdt = self.mdt_fc_out(mdt)
        mdt = self.relu(mdt)
        mdt = mdt.view(-1, len(self.devices['mdt'])*self.server_out_size)
        return mdt

    def ost_forward(self, ost):
        ost = ost.view(-1, ost.shape[-1])
        ost = self.ost_fc(ost)
        ost = self.relu(ost)
        ost = self.ost_fc_hidden(ost)
        ost = self.relu(ost)
        ost = self.ost_fc_out(ost)
        ost = self.relu(ost)
        ost = ost.view(-1, len(self.devices['ost'])*self.server_out_size)
        return ost

    def forward(self, mdt, ost):
        mdt = self.mdt_forward(mdt)
        ost = self.ost_forward(ost)
        x = torch.cat((mdt, ost), dim=1)
        x = self.fc_bridge(x)
        x = self.relu(x)
        #x = self.fc_hidden(x)
        #x = self.relu(x)
        x = self.fc_out(x)
        x = self.last_activation(x)
        return x



def get_workload_data_paths(config, workload, train=True):
    data_root = os.path.join(IOSENSE_ROOT, config['data']['output_dir'])
    sample_paths = {}
    timestamp_dirs = os.listdir(os.path.join(data_root, workload))
    # dirs are in the format of YYYY-MM-DD_HH-MM-SS
    # get the most recent timestamp_dir
    timestamp_dirs.sort()
    timestamp_dir = timestamp_dirs[-1]
    for file in os.listdir(os.path.join(data_root, workload, timestamp_dir)):
        if train:
            string_check = 'train'
        else:
            string_check = 'test'
        if string_check in file:
            # files are in the format of train_samples_[window_size].json
            window_size = file.split('_')[2]
            window_size = float(window_size.replace('.json', ''))
            sample_paths[window_size] = os.path.join(data_root, workload, timestamp_dir, file)
    return sample_paths
    

def get_data_paths(config):
    train_sample_paths = {}
    test_sample_paths = {}
    for workload in config['training']['train']['workloads']:
        train_sample_paths[workload] = get_workload_data_paths(config, workload, train=True)
    for workload in config['training']['test']['workloads']:
        test_sample_paths[workload] = get_workload_data_paths(config, workload, train=False)
    return train_sample_paths, test_sample_paths


def train_model(config):
    pass

def main():
    model_config = load_model_config()
    train_config = load_train_config()
    data_config = load_data_config()
    config = {"model_config": model_config,
              "train_config": train_config,
              "data_config": data_config
    }
    train_sample_paths, test_sample_paths = get_data_paths(config)
    train_samples = MetricsDataset(train_sample_paths, train=True, features=config['model_config']['model']['features'])
    training_scaler = train_samples.scaler
    devices = train_samples.devices
    train_loader = DataLoader(train_samples, batch_size=1, shuffle=True)
    test_samples = MetricsDataset(test_sample_paths, train=False, features=config['model_config']['model']['features'], scaler=training_scaler)
    validation_samples = test_samples[:int(0.8*len(test_samples))]
    test_samples = test_samples[int(0.8*len(test_samples)):]
    test_loader = DataLoader(test_samples, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_samples, batch_size=1, shuffle=True)
    model = SensitivityModel(devices,
                             config['model_config']['model']['features'],
                             hidden_size=config['model_config']['model']['hidden_size'], 
                             server_out_size=config['model_config']['model']['server_out_size'], 
                             output_size=config['model_config']['model']['output_size'], 
                             server_emb_size=config['model_config']['model']['server_emb_size'])

    
if __name__ == "__main__":
    main()