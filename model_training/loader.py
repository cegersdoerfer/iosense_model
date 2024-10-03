import argparse
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import re




def scale(mdt_features, ost_features, scaler=None, save_scaler=None):
    if scaler is None:
        mdt_scaler = StandardScaler()
        ost_scaler = StandardScaler()
        reshape_mdt = mdt_features.reshape(-1, mdt_features.shape[-1])
        reshape_ost = ost_features.reshape(-1, ost_features.shape[-1])
        mdt_scaler.fit(reshape_mdt)
        ost_scaler.fit(reshape_ost)
        if save_scaler:
            with open(f'{save_scaler}_mdt', 'wb') as f:
                pickle.dump(mdt_scaler, f)
            with open(f'{save_scaler}_ost', 'wb') as f:
                pickle.dump(ost_scaler, f)
        else:
            print("SCALER IS NOT SAVED AS PATH WAS NOT PROVIDED")
    else:
        # load the scaler
        with open(f'{scaler}_mdt', 'rb') as f:
            mdt_scaler = pickle.load(f)
        with open(f'{scaler}_ost', 'rb') as f:
            ost_scaler = pickle.load(f)
    for i in range(len(mdt_features)):
        mdt_features[i] = mdt_scaler.transform(mdt_features[i])
        ost_features[i] = ost_scaler.transform(ost_features[i])
    return ost_features, mdt_features, scaler

def get_devices(workload_data_sample):
    mdt_devices = []
    ost_devices = []
    for server_type in workload_data_sample['trace_features']:
        if server_type == 'mdt':
            server_string = 'mdt'
        else:
            server_string = 'ost'
        for feature in workload_data_sample['trace_features'][server_type]:
            # each feature will be [server_type]_[device_id]_[metric]
            device_id = re.search(f'({server_string}_\d+)', feature).group(1)
            if server_type == 'mdt' and device_id not in mdt_devices:
                mdt_devices.append(device_id)
            elif server_type == 'ost' and device_id not in ost_devices:
                ost_devices.append(device_id)
    devices = {'mdt': mdt_devices, 'ost': ost_devices}
    return devices
            
        



class MetricsDataset(Dataset):
    def __init__(self, workload_dirs, features, bin_thresholds=[2.0], train=True, scaler=None, augment=False, window_sizes=None):
        self.workload_dirs = workload_dirs
        self.features = features
        self.augment = augment
        self.num_bins = len(bin_thresholds) + 1
        self.bin_thresholds = bin_thresholds
        self.scaler = scaler
        self.train = train
        self.mdt_features = []
        self.ost_features = []
        self.target = []

        self.load_data(workload_dirs, window_sizes)
        
    def load_data(self, workload_dirs, window_sizes):
        total_idx = 0
        for workload in workload_dirs:
            for window_size in workload_dirs[workload]:
                if window_sizes is not None:
                    if window_size not in window_sizes:
                        continue
                
                # load the data file for the current window size
                print(f"Loading data from {workload_dirs[workload][window_size]}")
                workload_data = []
                for file_path in workload_dirs[workload][window_size]:
                    with open(file_path, 'r') as f:
                        samples_list = json.load(f)
                    workload_data.extend(samples_list)
                
                self.devices = get_devices(workload_data[0])
                print(f"Devices: {self.devices}")

                for sample in workload_data:
                    # each sample should have a trace_features, stats_features, absolute_runtime_diff, relative_runtime_diff
                    # trace_features also has 'ost' and 'mdt'
                    self.mdt_features.append([])
                    self.ost_features.append([])
                    for target_device in self.devices['ost']:
                        self.ost_features[-1].append([])
                        for feature in self.features['ost_trace']:
                            try:
                                self.ost_features[-1][-1].append(sample['trace_features']['ost'][f'{target_device}_{feature}'])
                            except:
                                self.ost_features[-1][-1].append(0)
                        for feature in self.features['stats']:
                            try:
                                self.ost_features[-1][-1].append(sample['stats_features']['ost'][f'{target_device}_{feature}'])
                            except:
                                self.ost_features[-1][-1].append(0)
                        self.ost_features[-1][-1] = np.nan_to_num(self.ost_features[-1][-1], nan=0)
                    for target_device in self.devices['mdt']:
                        self.mdt_features[-1].append([])
                        for feature in self.features['mdt_trace']:
                            try:
                                self.mdt_features[-1][-1].append(sample['trace_features']['mdt'][f'{target_device}_{feature}'])
                            except:
                                self.mdt_features[-1][-1].append(0)
                        for feature in self.features['stats']:
                            try:
                                self.mdt_features[-1][-1].append(sample['stats_features']['mdt'][f'{target_device}_{feature}'])
                            except:
                                self.mdt_features[-1][-1].append(0)
                        self.mdt_features[-1][-1] = np.nan_to_num(self.mdt_features[-1][-1], nan=0)
                    

                    self.mdt_features[-1] = np.array(self.mdt_features[-1])
                    self.ost_features[-1] = np.array(self.ost_features[-1])
                    self.target.append(sample['relative_runtime_diff'])
                    total_idx += 1
                    
                    #pass
                    if self.augment:
                        for i in range(len(self.devices['ost'])-1):
                            # rotate positions of OST servers
                            self.mdt_features.append(self.mdt_features[-1])
                            self.ost_features.append(np.roll(self.ost_features[-1], 1, axis=0))
                            self.target.append(sample['relative_runtime_diff'])
                            total_idx += 1
                            
                            
        print(f"Total number of data points: {total_idx}")

        self.target = np.array(self.target)
        self.target = np.digitize(self.target, self.bin_thresholds)
        self.target = np.eye(self.num_bins)[self.target]
        self.ost_features = np.array(self.ost_features)
        self.mdt_features = np.array(self.mdt_features)
        print(f"ost_features shape: {self.ost_features.shape}")
        print(f"mdt_features shape: {self.mdt_features.shape}")
        self.ost_features = np.nan_to_num(self.ost_features, nan=0)
        self.mdt_features = np.nan_to_num(self.mdt_features, nan=0)
        if self.num_bins == 2:
            # if only 2 bins, then we can convert the target to a single column
            self.target = np.argmax(self.target, axis=1)
            self.target = self.target.reshape(-1, 1)
        else:
            self.target = self.target.reshape(-1, self.num_bins)
        print(f"sum of target: {np.sum(self.target, axis=0)}")
        
        print(f"target shape: {self.target.shape}")
       # print(f"target: {self.target}")
        print(f"postive samples: {np.sum(self.target)}")
        print(f"negative samples: {len(self.target) - np.sum(self.target)}")
        self.ost_features, self.mdt_features, self.scaler = scale(self.mdt_features, self.ost_features, self.scaler, save_scaler='scaler' if self.train else None)
        self.plot_data()

    def plot_data(self):
        plt.figure()
        plt.hist(self.target, bins=self.num_bins)
        plt.savefig('target_distribution.png')
        plt.close()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.mdt_features[idx], self.ost_features[idx], self.target[idx]


if __name__ == "__main__":
    # test the dataset
    workload_dir = ['/Users/chris/Downloads/darshan-analysis/python-files/data_files/dlio_bench_unet3d', '/Users/chris/Downloads/darshan-analysis/python-files/data_files/dlio_bench_bert']
    dataset = MetricsDataset(workload_dir, train=True)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
    dataset = MetricsDataset(workload_dir, train=False)
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)

