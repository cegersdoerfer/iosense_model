import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.config import load_model_config, load_data_config, load_train_config, IOSENSE_ROOT








class SensitivityModel(nn.Module):
    def __init__(self, hidden_size=16, server_out_size = 8, output_size=1, server_emb_size=32):
        super(SensitivityModel, self).__init__()
        mdt_input_width = len(SERVER_COLUMNS) + len(MDT_TRACE_KEYS) + 1
        print('mdt_input_width: ', mdt_input_width)
        ost_input_width = len(SERVER_COLUMNS) + len(OST_TRACE_KEYS) + 1
        print('ost_input_width: ', ost_input_width)
        self.server_out_size = server_out_size
        self.mdt_fc = nn.Linear(mdt_input_width, server_emb_size)
        self.mdt_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.mdt_fc_out = nn.Linear(hidden_size, server_out_size)
        
        self.ost_fc = nn.Linear(ost_input_width, server_emb_size)
        self.ost_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.ost_fc_out = nn.Linear(hidden_size, server_out_size)
        self.fc_bridge = nn.Linear(len(MDT_SERVERS)*server_out_size + len(OST_SERVERS)*server_out_size, hidden_size)
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
        mdt = mdt.view(-1, len(MDT_SERVERS)*self.server_out_size)
        return mdt

    def ost_forward(self, ost):
        ost = ost.view(-1, ost.shape[-1])
        ost = self.ost_fc(ost)
        ost = self.relu(ost)
        ost = self.ost_fc_hidden(ost)
        ost = self.relu(ost)
        ost = self.ost_fc_out(ost)
        ost = self.relu(ost)
        ost = ost.view(-1, len(OST_SERVERS)*self.server_out_size)
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


def load_samples(sample_paths):
    pass


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



def main():
    model_config = load_model_config()
    train_config = load_train_config()
    data_config = load_data_config()
    config = {"model": model_config,
              "training": train_config,
              "data": data_config
    }
    train_sample_paths, test_sample_paths = get_data_paths(config)
    train_samples = load_samples(train_sample_paths)
    test_samples = load_samples(test_sample_paths)
    model = SensitivityModel(hidden_size=config['model']['hidden_size'], 
                             server_out_size=config['model']['server_out_size'], 
                             output_size=config['model']['output_size'], 
                             server_emb_size=config['model']['server_emb_size'])
if __name__ == "__main__":
    main()