import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.config import load_model_config, load_data_config, load_train_config, IOSENSE_ROOT
from model_training.loader import MetricsDataset




class SensitivityModel(nn.Module):
    def __init__(self, devices, features, hidden_size=16, global_hidden_size=16, server_out_size = 8, output_size=1, server_emb_size=16):
        self.devices = devices
        self.features = features
        super(SensitivityModel, self).__init__()
        mdt_input_width = len(features['stats']) + len(features['mdt_trace'])
        print('mdt_input_width: ', mdt_input_width)
        ost_input_width = len(features['stats']) + len(features['ost_trace'])
        print('ost_input_width: ', ost_input_width)
        self.server_out_size = server_out_size
        self.mdt_fc = nn.Linear(mdt_input_width, server_emb_size)
        print('mdt_fc: ', self.mdt_fc)
        self.mdt_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.mdt_fc_out = nn.Linear(hidden_size, server_out_size)
        
        self.ost_fc = nn.Linear(ost_input_width, server_emb_size)
        self.ost_fc_hidden = nn.Linear(server_emb_size, hidden_size)
        self.ost_fc_out = nn.Linear(hidden_size, server_out_size)
        self.fc_bridge = nn.Linear(len(devices['mdt'])*server_out_size + len(devices['ost'])*server_out_size, global_hidden_size)
        #self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(global_hidden_size, output_size)
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
    data_root = os.path.join(IOSENSE_ROOT, config['data_config']['output_dir'])
    sample_paths = {}
    timestamp_dirs = os.listdir(os.path.join(data_root, workload))
    dirs_list = []
    if train:
        set_string = 'train'
    else:
        set_string = 'train'
    if 'load_setting' in config['train_config'][set_string]:
        load_setting = config['train_config'][set_string]['load_setting']
    else:
        load_setting = 'most_recent'
    print('load_setting: ', load_setting)
    if load_setting == 'most_recent':
        # dirs are in the format of YYYY-MM-DD_HH-MM-SS
        # get the most recent timestamp_dir
        timestamp_dirs.sort()
        timestamp_dir = timestamp_dirs[-1]
        dirs_list.append(timestamp_dir)
    else:
        dirs_list = timestamp_dirs

    for timestamp_dir in dirs_list:
        for file in os.listdir(os.path.join(data_root, workload, timestamp_dir)):
            if train:
                string_check = 'train'
            else:
                string_check = 'train'
            if string_check in file:
                # files are in the format of train_samples_[window_size].json
                window_size = file.split('_')[2]
                window_size = float(window_size.replace('.json', ''))
                if window_size not in sample_paths:
                    sample_paths[window_size] = []
                sample_paths[window_size].append(os.path.join(data_root, workload, timestamp_dir, file))
    return sample_paths
    

def get_data_paths(config):
    train_sample_paths = {}
    test_sample_paths = {}
    for workload in config['train_config']['train']['workloads']:
        train_sample_paths[workload] = get_workload_data_paths(config, workload, train=True)
    for workload in config['train_config']['test']['workloads']:
        test_sample_paths[workload] = get_workload_data_paths(config, workload, train=False)
    return train_sample_paths, test_sample_paths



def get_metrics(output, label, metrics, num_bins=2):
    if num_bins == 2:
        metrics['tp'] += ((output > 0.5) & (label == 1)).sum().item()
        metrics['fp'] += ((output > 0.5) & (label == 0)).sum().item()
        metrics['tn'] += ((output <= 0.5) & (label == 0)).sum().item()
        metrics['fn'] += ((output <= 0.5) & (label == 1)).sum().item()
        metrics['acc'] = (metrics['tp'] + metrics['tn']) / (metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'])
        if metrics['tp'] + metrics['fp'] > 0:
            metrics['prec'] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
        else:
            metrics['prec'] = 0
        if metrics['tp'] + metrics['fn'] > 0:
            metrics['rec'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
        else:
            metrics['rec'] = 0
        if metrics['prec'] + metrics['rec'] > 0:
            metrics['f1'] = 2 * (metrics['prec'] * metrics['rec']) / (metrics['prec'] + metrics['rec'])
        else:
            metrics['f1'] = 0
    else:
        metrics['tp'] += (output == label).sum().item()
        metrics['fp'] += ((output == 0) & (label == 1)).sum().item()
        metrics['tn'] += ((output == 0) & (label == 0)).sum().item()
        metrics['fn'] += ((output == 1) & (label == 0)).sum().item()
        metrics['acc'] = (metrics['tp'] + metrics['tn']) / (metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'])
        if metrics['tp'] + metrics['fp'] > 0:
            metrics['prec'] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
        else:
            metrics['prec'] = 0
        if metrics['tp'] + metrics['fn'] > 0:
            metrics['rec'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
        else:
            metrics['rec'] = 0
        if metrics['prec'] + metrics['rec'] > 0:
            metrics['f1'] = 2 * (metrics['prec'] * metrics['rec']) / (metrics['prec'] + metrics['rec'])
        else:
            metrics['f1'] = 0
    return metrics


def train_model(train_data_loader, validation_data_loader, model, num_bins=2):
    if num_bins == 2:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_losses = []
    valid_losses = []
    train_f1 = []
    valid_f1 = []
    best_model = model
    best_loss = float('inf')
    loss_steps = 5000
    num_epochs = 30
    train_loss = 0
    idx = 0
    for epoch in range(num_epochs):
        model.train()
        train_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        for mdt, ost, label in train_data_loader:
            # Convert input data to Float
            mdt = mdt.float()
            ost = ost.float()
            label = label.float()
            optimizer.zero_grad()
            output = model(mdt, ost)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_metrics = get_metrics(output, label, train_metrics, num_bins=num_bins)
            if idx % loss_steps == 0:   
                print(f"Epoch {epoch+1}/{num_epochs}, Step {idx+1}/{len(train_data_loader)}, Loss: {train_loss/loss_steps:.4f}")
                train_loss = 0
            idx += 1
        print(f"Training metrics: {train_metrics}")
        train_losses.append(train_loss)
        train_f1.append(train_metrics['f1'])

        model.eval()
        valid_loss = 0
        valid_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        with torch.no_grad():
            for mdt, ost, label in validation_data_loader:
                mdt = mdt.float()
                ost = ost.float()
                label = label.float()
                output = model(mdt, ost)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_metrics = get_metrics(output, label, valid_metrics, num_bins=num_bins)
        valid_losses.append(valid_loss)
        valid_f1.append(valid_metrics['f1'])
        print(f"Validation metrics: {valid_metrics}\n\n")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model
    return train_losses, train_f1, valid_losses, valid_f1, best_model, criterion

def test_model(test_data_loader, model, criterion):
    model.eval()
    test_loss = 0
    test_metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    with torch.no_grad():
        for mdt, ost, label in test_data_loader:
            # Convert input data to Float
            mdt = mdt.float()
            ost = ost.float()
            label = label.float()
            output = model(mdt, ost)
            loss = criterion(output, label)
            test_loss += loss.item()
            test_metrics = get_metrics(output, label, test_metrics, num_bins=2)
        print(f"Test metrics: {test_metrics}")
    return test_loss, test_metrics


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def main():
    model_config = load_model_config()
    train_config = load_train_config()
    data_config = load_data_config()
    config = {"model_config": model_config,
              "train_config": train_config,
              "data_config": data_config
    }
    train_sample_paths, test_sample_paths = get_data_paths(config)
    train_samples = MetricsDataset(train_sample_paths, train=True, features=config['model_config']['features'], window_sizes=config['train_config']['train']['window_sizes'])
    training_scaler = train_samples.scaler
    devices = train_samples.devices
    train_loader = DataLoader(train_samples, batch_size=32, shuffle=True)
    print('train set size: ', len(train_samples))
    test_samples = MetricsDataset(test_sample_paths, train=False, features=config['model_config']['features'], scaler=training_scaler, window_sizes=config['train_config']['test']['window_sizes'])
    validation_size = int(0.8*len(test_samples))
    test_size = len(test_samples) - validation_size
    test_dataset, validation_dataset = torch.utils.data.random_split(test_samples, [test_size, validation_size])
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    print('validation set size: ', len(validation_dataset))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    model = SensitivityModel(devices,
                             config['model_config']['features'],
                             hidden_size=config['model_config']['hidden_size'], 
                             server_out_size=config['model_config']['server_out_size'], 
                             output_size=config['model_config']['output_size'], 
                             server_emb_size=config['model_config']['server_emb_size'])
    print(model)
    
    train_losses, train_f1, valid_losses, valid_f1, best_model, criterion = train_model(train_loader, validation_loader, model)
    test_losses, test_f1 = test_model(test_loader, best_model, criterion)

    print(f"Train Loss: {train_losses[-1]}, Train F1: {train_f1[-1]}")
    print(f"Validation Loss: {valid_losses[-1]}, Validation F1: {valid_f1[-1]}")
    print(f"Test Loss: {test_losses}, Test F1: {test_f1}")

    save_model(best_model, "best_model.pth")

    

    
if __name__ == "__main__":
    main()