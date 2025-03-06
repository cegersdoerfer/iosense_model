import json
import os
import pandas as pd
import numpy as np
import random
import subprocess
from data_processor.parse_darshan_txt import parse_darshan_txt
import re
import argparse
from utils.config import load_cluster_config, IOSENSE_ROOT, load_data_config



def load_darshan_trace_from_dir(dir_path, config_name, run_txt, devices):
    print(f"Loading Darshan traces from {dir_path} for config {config_name}...")
    traces = []
    file_ids_segments_offsets_map = {}
    retry_files = []
    for file in os.listdir(dir_path):
        if not run_txt:
            if file.endswith('.darshan') and config_name in file:
                print(f"Processing file: {file}")
                txt_file = file.replace('.darshan', '.txt')
                command = f"darshan-dxt-parser --show-incomplete {os.path.join(dir_path, file)} > {os.path.join(dir_path, txt_file)}"
                subprocess.run(command, shell=True)
                with open(os.path.join(dir_path, txt_file), 'r') as f:
                    darshan_txt = f.read()
                trace_df, trace_start_time, trace_runtime, file_ids_segments_offsets_map, retry_at_end = parse_darshan_txt(darshan_txt, devices, file_ids_segments_offsets_map)
                if retry_at_end:
                    retry_files.append(file)
                if trace_df is not None:
                    traces.append(trace_df)
        else:
            if file.endswith('.txt') and config_name in file:
                print(f"Processing file: {file}")
                with open(os.path.join(dir_path, file), 'r') as f:
                    trace_txt = f.read()
                trace_df, trace_start_time, trace_runtime, file_ids_segments_offsets_map, retry_at_end = parse_darshan_txt(trace_txt, devices, file_ids_segments_offsets_map)
                if retry_at_end:
                    retry_files.append(file)
                if trace_df is not None:
                    traces.append(trace_df)
    if len(retry_files) > 0:
        print(f"Retrying {len(retry_files)} files...")
        for file in retry_files:
            print(f"Retrying file: {file}")
            if file.endswith('.darshan'):
                txt_file = file.replace('.darshan', '.txt')
                command = f"darshan-dxt-parser --show-incomplete {os.path.join(dir_path, file)} > {os.path.join(dir_path, txt_file)}"
                subprocess.run(command, shell=True)
                with open(os.path.join(dir_path, txt_file), 'r') as f:
                    darshan_txt = f.read()
                trace_df, trace_start_time, trace_runtime, file_ids_segments_offsets_map, _ = parse_darshan_txt(darshan_txt, devices, file_ids_segments_offsets_map)
                if trace_df is not None:
                    traces.append(trace_df)
    if traces:
        darshan_df = pd.concat(traces)
        darshan_df = darshan_df.sort_values(by=['start'])
        darshan_df.reset_index(inplace=True)
        darshan_df = darshan_df.drop(columns=['index'])
        print("Darshan traces loaded and concatenated.")
        print(darshan_df.head())
        return darshan_df
    else:
        print("No Darshan traces found.")
        return None

def process_stats_file(file_path):
    # stats file will have columns: time_stamp, major, minor, device_name, and read_ios, read_merges, sectors_read, read_ticks, write_ios, write_merges, sectors_written, write_ticks, in_flight, io_ticks, time_in_queue, discard_ios, discard_merges, discard_sectors, discard_ticks
    # ticks are milliseconds as outlined by https://www.kernel.org/doc/Documentation/block/stat.txt
    # time_stamp is in '%Y-%m-%d %H:%M:%S.%3N' format
    # stats reset at 32 bit max
    # example of a line:
    # 2024-09-24 22:13:22.050    8      21 sdb5 29916 0 246718 4430 365325 656618 370340112 3908352 0 394904 3918091 5444 0 891432776 5307
    columns=['time_stamp', 'major', 'minor', \
                'device_name', 'read_ios', 'read_merges', 'sectors_read', 'read_ticks', \
                'write_ios', 'write_merges', 'sectors_written', 'write_ticks', 'in_flight', \
                'io_ticks', 'time_in_queue', 'discard_ios', 'discard_merges', 'discard_sectors', \
                'discard_ticks']
    data = {column: [] for column in columns}
    # read in the txt file
    with open(file_path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            line = line.split()
            data['time_stamp'].append(line[0] + ' ' + line[1])
            data['major'].append(line[2])
            data['minor'].append(line[3])
            data['device_name'].append(line[4])
            start_index = 4
            for column_idx, column in enumerate(columns[start_index:]):
                shifted_column_idx = column_idx + start_index + 1
                if line_idx == 0:
                    data[column].append(int(line[shifted_column_idx]))
                else:
                    # for in_flight, do not calculate difference
                    if column == 'in_flight':
                        data[column].append(int(line[shifted_column_idx]))
                    else:
                        if int(line[shifted_column_idx]) < int(data[column][line_idx - 1]):
                            data[column].append(int(line[shifted_column_idx]) + (2**32 - int(data[column][line_idx - 1])))
                        else:
                            data[column].append(int(line[shifted_column_idx]) - int(data[column][line_idx - 1]))
        for column_idx, column in enumerate(columns[start_index:]):
            shifted_column_idx = column_idx + start_index
            data[column][0] = 0
    lengths = [len(data[col]) for col in columns]
    if len(set(lengths)) != 1:
        raise ValueError(f"All arrays must be of the same length. Current lengths: {lengths}")
    df = pd.DataFrame(data)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.sort_values(by=['time_stamp'], inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    print(f"Stats file {file_path} processed.")
    # show summary statistics
    print(df.describe())
    return df

def load_stats_from_dir(dir_path):
    print(f"Loading stats from directory: {dir_path}")
    stats = {}
    for folder in os.listdir(dir_path):
        print(f"Processing folder: {folder}")
        if not os.path.isdir(os.path.join(dir_path, folder)):
            print(f"Skipping {folder} as it's not a directory")
            continue
        else:
            subdirs = os.listdir(os.path.join(dir_path, folder))
            print(f"Found {len(subdirs)} subdirectories in {folder}")
            for subdir in subdirs:
                if any([x in subdir for x in ['OST', 'MDT']]):
                    # dirs will be in fsname_[MDT|OST][xxxx].log, like hasanfs-MDT0000.log, extract MDT or OST and hex number in xxxx
                    match = re.match(r'.*?-((?:MDT|OST)([0-9A-Fa-f]{4}))\.log', subdir)
                    if match:
                        print(f"Found match: {match.group(0)}")
                        id = match.group(1)
                        hex_num = match.group(2)
                        if 'MDT' in id:
                            id = str(int(hex_num, 16))  # Convert hex to decimal
                            type_string = 'mdt'
                        else:
                            id = str(int(hex_num, 16))  # Convert hex to decimal
                            type_string = 'ost'
                        id = f'{type_string}_{id}'

                        print(f"Processing subdir: {subdir}, ID: {id}")
                        if id not in stats:
                            stats[id] = []
                            print(f"Created new list for {id}")
                        file_path = os.path.join(dir_path, folder, subdir)
                        print(f"Processing stats file: {file_path}")
                        stats[id].append(process_stats_file(file_path))
                        print(f"Processed stats file for {id}")
                else:
                    print(f"Skipping {subdir} as it doesn't match OST or MDT pattern")
    print(f"Processing complete. Found data for {len(stats)} devices.")
    for id in stats:
        print(f"Concatenating and sorting data for {id}")
        stats[id] = pd.concat(stats[id])
        stats[id]['time_stamp'] = pd.to_datetime(stats[id]['time_stamp'], format='%Y-%m-%d %H:%M:%S.%f')
        stats[id].sort_values(by=['time_stamp'], inplace=True)
        stats[id].reset_index(inplace=True)
        print(f"Processed {len(stats[id])} rows for {id}")
    print("Stats loaded, concatenated, and sorted.")
    return stats
            

def get_data(model_config, run_txt):
    print("Getting data based on model configuration...")
    data = {'baseline_traces': {}, 'interference_traces': {}}
    data_types = ['stats', 'darshan_logs']
    workload = model_config['workload']
    time_stamp_dir = model_config['time_stamp_dir']
    data_dir = os.path.join(IOSENSE_ROOT, "data")
    devices = {'mdt': [], 'ost': []}
    for data_type in data_types:
        if data_type == 'darshan_logs':
            interference_levels = os.listdir(os.path.join(data_dir, workload, data_type, time_stamp_dir))
            for interference_level in interference_levels:
                interference_level_num = int(interference_level.split('_')[-1])
                interference_level_files = os.listdir(os.path.join(data_dir, workload, data_type, time_stamp_dir, interference_level))
                for repitition in interference_level_files:
                    repitition_files = os.listdir(os.path.join(data_dir, workload, data_type, time_stamp_dir, interference_level, repitition))
                    configs = []
                    for file in repitition_files:
                        if file.endswith('.darshan'):
                            # file will be in format configname_idx.darshan like debug_config_0.darshan
                            print(f"Processing file: {file}")
                            config_name = re.match(r'(.*?)_[0-9].darshan', file).group(1)
                            if config_name not in configs:
                                configs.append(config_name)
                    for config in configs:
                        print("processing config: ", config)
                        trace_df = load_darshan_trace_from_dir(os.path.join(data_dir, workload, data_type, time_stamp_dir, interference_level, repitition), config, run_txt, devices)
                        if trace_df is None:
                            continue

                        if interference_level_num == 0:
                            data['baseline_traces'][config] = trace_df
                        else:
                            if interference_level_num not in data['interference_traces']:
                                data['interference_traces'][interference_level_num] = {}
                            if repitition not in data['interference_traces'][interference_level_num]:
                                data['interference_traces'][interference_level_num][repitition] = {}
                            data['interference_traces'][interference_level_num][repitition][config] = trace_df
        elif data_type == 'stats':
            data['stats'] = load_stats_from_dir(os.path.join(data_dir, workload, data_type, time_stamp_dir))
            devices['mdt'] = [device for device in data['stats'] if 'mdt' in device]
            devices['ost'] = [device for device in data['stats'] if 'ost' in device]

            if devices['mdt'] == [] or devices['ost'] == []:
                raise ValueError('No MDT or OST devices found')
    
    print("Data retrieval complete.")
    return data, devices



def get_trace_features(trace_df_window, devices):
    window_runtime = trace_df_window['end'].max() - trace_df_window['start'].min()

    trace_features = {'ost': {}, 'mdt': {}}
    ost_devices = devices['ost']
    mdt_devices = devices['mdt']
    for ost_device in ost_devices:
        ost_device_df = trace_df_window[trace_df_window[ost_device]==1]
        ost_window_runtime = ost_device_df['end'].max() - ost_device_df['start'].min() if len(ost_device_df) > 0 else 0
        
        # Calculate total time spent on I/O operations for this device
        io_time = 0
        if len(ost_device_df) > 0:
            # Sort operations by start time
            sorted_ops = ost_device_df.sort_values(by='start')
            current_end = sorted_ops.iloc[0]['start']
            
            # Merge overlapping operations to get actual I/O time
            for _, op in sorted_ops.iterrows():
                if op['start'] > current_end:
                    # Gap between operations
                    io_time += 0  # No additional time for gap
                    current_end = op['end']
                elif op['end'] > current_end:
                    # Partial overlap or extension
                    io_time += op['end'] - current_end
                    current_end = op['end']
                # Completely overlapping operations don't add time
        
        # Calculate idle time
        idle_time = ost_window_runtime - io_time if ost_window_runtime > 0 else 0
        trace_features['ost'][f'{ost_device}_idle_time'] = idle_time
        trace_features['ost'][f'{ost_device}_idle_time_percentage'] = (idle_time / ost_window_runtime * 100) if ost_window_runtime > 0 else 0

        
        # Existing code
        trace_features['ost'][f'{ost_device}_num_read_ops'] = len(trace_df_window[(trace_df_window[ost_device] == 1) & (trace_df_window['operation'] == 'read')])
        trace_features['ost'][f'{ost_device}_num_write_ops'] = len(trace_df_window[(trace_df_window[ost_device] == 1) & (trace_df_window['operation'] == 'write')])
        trace_features['ost'][f'{ost_device}_num_ops'] = trace_features['ost'][f'{ost_device}_num_read_ops'] + trace_features['ost'][f'{ost_device}_num_write_ops']
        if io_time > 0:
            print(f"ost_window_runtime: {ost_window_runtime}")
            print(f"io_time: {io_time}")
            trace_features['ost'][f'{ost_device}_num_read_ops_per_sec'] = trace_features['ost'][f'{ost_device}_num_read_ops'] / io_time
            trace_features['ost'][f'{ost_device}_num_write_ops_per_sec'] = trace_features['ost'][f'{ost_device}_num_write_ops'] / io_time
            trace_features['ost'][f'{ost_device}_num_ops_per_sec'] = trace_features['ost'][f'{ost_device}_num_ops'] / io_time
        else:
            trace_features['ost'][f'{ost_device}_num_read_ops_per_sec'] = 0
            trace_features['ost'][f'{ost_device}_num_write_ops_per_sec'] = 0
            trace_features['ost'][f'{ost_device}_num_ops_per_sec'] = 0
        trace_features['ost'][f'{ost_device}_size_read_ops'] = trace_df_window[(trace_df_window[ost_device] == 1) & (trace_df_window['operation'] == 'read')]['size'].sum()
        trace_features['ost'][f'{ost_device}_size_write_ops'] = trace_df_window[(trace_df_window[ost_device] == 1) & (trace_df_window['operation'] == 'write')]['size'].sum()
        trace_features['ost'][f'{ost_device}_size_ops'] = trace_features['ost'][f'{ost_device}_size_read_ops'] + trace_features['ost'][f'{ost_device}_size_write_ops']
        if io_time > 0:
            print(f"ost_window_runtime: {ost_window_runtime}")
            print(f"io_time: {io_time}")
            trace_features['ost'][f'{ost_device}_size_read_ops_per_sec'] = trace_features['ost'][f'{ost_device}_size_read_ops'] / io_time
            trace_features['ost'][f'{ost_device}_size_write_ops_per_sec'] = trace_features['ost'][f'{ost_device}_size_write_ops'] / io_time
            trace_features['ost'][f'{ost_device}_size_ops_per_sec'] = trace_features['ost'][f'{ost_device}_size_ops'] / io_time
        else:
            trace_features['ost'][f'{ost_device}_size_read_ops_per_sec'] = 0
            trace_features['ost'][f'{ost_device}_size_write_ops_per_sec'] = 0
            trace_features['ost'][f'{ost_device}_size_ops_per_sec'] = 0
    for mdt_device in mdt_devices:
        mdt_device_df = trace_df_window[trace_df_window[mdt_device]==1]
        mdt_window_runtime = mdt_device_df['end'].max() - mdt_device_df['start'].min() if len(mdt_device_df) > 0 else 0
        
        # Calculate total time spent on I/O operations for this device
        io_time = 0
        if len(mdt_device_df) > 0:
            # Sort operations by start time
            sorted_ops = mdt_device_df.sort_values(by='start')
            current_end = sorted_ops.iloc[0]['start']
            
            # Merge overlapping operations to get actual I/O time
            for _, op in sorted_ops.iterrows():
                if op['start'] > current_end:
                    # Gap between operations
                    io_time += 0  # No additional time for gap
                    current_end = op['end']
                elif op['end'] > current_end:
                    # Partial overlap or extension
                    io_time += op['end'] - current_end
                    current_end = op['end']
                # Completely overlapping operations don't add time
        
        # Calculate idle time
        idle_time = mdt_window_runtime - io_time if mdt_window_runtime > 0 else 0
        trace_features['mdt'][f'{mdt_device}_idle_time'] = idle_time
        trace_features['mdt'][f'{mdt_device}_idle_time_percentage'] = (idle_time / mdt_window_runtime * 100) if mdt_window_runtime > 0 else 0
        
        # Existing code
        trace_features['mdt'][f'{mdt_device}_num_stat_ops'] = len(trace_df_window[(trace_df_window[mdt_device] == 1) & (trace_df_window['operation'] == 'stat')])
        trace_features['mdt'][f'{mdt_device}_num_open_ops'] = len(trace_df_window[(trace_df_window[mdt_device] == 1) & (trace_df_window['operation'] == 'open')])
        trace_features['mdt'][f'{mdt_device}_num_close_ops'] = len(trace_df_window[(trace_df_window[mdt_device] == 1) & (trace_df_window['operation'] == 'close')])
        trace_features['mdt'][f'{mdt_device}_num_ops'] = trace_features['mdt'][f'{mdt_device}_num_stat_ops'] + trace_features['mdt'][f'{mdt_device}_num_open_ops'] + trace_features['mdt'][f'{mdt_device}_num_close_ops']
        if io_time > 0:
            trace_features['mdt'][f'{mdt_device}_num_ops_per_sec'] = trace_features['mdt'][f'{mdt_device}_num_ops'] / io_time
            trace_features['mdt'][f'{mdt_device}_num_stat_ops_per_sec'] = trace_features['mdt'][f'{mdt_device}_num_stat_ops'] / io_time
            trace_features['mdt'][f'{mdt_device}_num_open_ops_per_sec'] = trace_features['mdt'][f'{mdt_device}_num_open_ops'] / io_time
            trace_features['mdt'][f'{mdt_device}_num_close_ops_per_sec'] = trace_features['mdt'][f'{mdt_device}_num_close_ops'] / io_time
        else:
            trace_features['mdt'][f'{mdt_device}_num_ops_per_sec'] = 0
            trace_features['mdt'][f'{mdt_device}_num_stat_ops_per_sec'] = 0
            trace_features['mdt'][f'{mdt_device}_num_open_ops_per_sec'] = 0
            trace_features['mdt'][f'{mdt_device}_num_close_ops_per_sec'] = 0
    return trace_features, window_runtime

def get_stats_features(stats_df_window, time_window_size):
    stats_features = {'mdt': {}, 'ost': {}}
    devices = list(stats_df_window.keys())
    for device in devices:
        total_read_ios = stats_df_window[device]['read_ios'].sum()
        total_write_ios = stats_df_window[device]['write_ios'].sum()
        sectors_read = stats_df_window[device]['sectors_read'].sum()
        sectors_written = stats_df_window[device]['sectors_written'].sum()
        read_ticks = stats_df_window[device]['read_ticks'].sum()
        write_ticks = stats_df_window[device]['write_ticks'].sum()
        time_in_queue = stats_df_window[device]['time_in_queue'].sum()
        if 'mdt' in device:
            device_type = 'mdt'
        else:
            device_type = 'ost'
        stats_features[device_type][f'{device}_read_throughput'] = sectors_read / time_window_size
        stats_features[device_type][f'{device}_write_throughput'] = sectors_written / time_window_size
        stats_features[device_type][f'{device}_read_iops'] = total_read_ios / time_window_size
        stats_features[device_type][f'{device}_write_iops'] = total_write_ios / time_window_size
        stats_features[device_type][f'{device}_read_ticks'] = read_ticks
        stats_features[device_type][f'{device}_write_ticks'] = write_ticks
        stats_features[device_type][f'{device}_time_in_queue'] = time_in_queue
    return stats_features

def get_label(baseline_trace_df_window, trace_df_runtime):
    baseline_runtime = baseline_trace_df_window['end'].max() - baseline_trace_df_window['start'].min()
    absolute_runtime_diff = trace_df_runtime - baseline_runtime
    if baseline_runtime > 0:
        relative_runtime_diff = absolute_runtime_diff / baseline_runtime
    else:
        print('baseline runtime is 0')
        relative_runtime_diff = 0
    return absolute_runtime_diff, relative_runtime_diff

    

def calculate_sample(trace_df_window, baseline_trace_df_window, stats_df_window, time_window_size, devices):
    stats_features = get_stats_features(stats_df_window, time_window_size)
    trace_features, window_runtime = get_trace_features(trace_df_window, devices)
    if window_runtime > 0:
        time_window_size = window_runtime
    absolute_runtime_diff, relative_runtime_diff = get_label(baseline_trace_df_window, window_runtime)
    sample = {
        'trace_features': trace_features,
        'stats_features': stats_features,
        'absolute_runtime_diff': absolute_runtime_diff,
        'relative_runtime_diff': relative_runtime_diff
    }
    return sample

def create_samples(data, time_window_size, test_size, devices):
    print(f"Creating samples with time window size: {time_window_size}")
    train_samples = []
    test_samples = []
    
    for interference_level in data['interference_traces']:
        for repitition in data['interference_traces'][interference_level]:
            for config in data['interference_traces'][interference_level][repitition]:
                trace_df = data['interference_traces'][interference_level][repitition][config]
                baseline_trace_df = data['baseline_traces'][config]
                trace_start_time = trace_df['start'].min()
                print(f"trace_start_time: {pd.Timestamp.fromtimestamp(trace_start_time)}")
                trace_end_time = trace_df['end'].max()
                print(f"trace_end_time: {pd.Timestamp.fromtimestamp(trace_end_time)}")
                num_windows = int((trace_end_time - trace_start_time) / time_window_size)
                print(f"num_windows: {num_windows}")
                for i in range(num_windows):
                    start_time = trace_start_time + i * time_window_size
                    end_time = trace_start_time + (i + 1) * time_window_size
                    #print(f"Window {i}: {pd.Timestamp.fromtimestamp(start_time)} to {pd.Timestamp.fromtimestamp(end_time)}")
                    trace_df_window = trace_df[(trace_df['start'] >= start_time) & (trace_df['start'] < end_time)]
                    if len(trace_df_window) == 0:
                        continue
                    # get the same operation indices in the baseline trace
                    baseline_trace_df_window = baseline_trace_df.iloc[trace_df_window.index[0]:trace_df_window.index[-1]+1]
                    # get the stats for the same time window
                    stats_df_window = {}
                    # convert start_time and end_time to pd.Timestamp
                    start_time = pd.Timestamp.fromtimestamp(start_time)
                    end_time = pd.Timestamp.fromtimestamp(end_time)
                    #print(f"start_time: {start_time}, end_time: {end_time}")
                    for device in data['stats']:
                        #print(f"min time_stamp: {data['stats'][device]['time_stamp'].min()}, max time_stamp: {data['stats'][device]['time_stamp'].max()}")
                        stats_df_window[device] = data['stats'][device][(data['stats'][device]['time_stamp'] >= start_time) & (data['stats'][device]['time_stamp'] < end_time)]
                        #if len(stats_df_window[device]) != 0:
                        #print(f"stats_df_window[device]: {stats_df_window[device]}")
                    sample = calculate_sample(trace_df_window, baseline_trace_df_window, stats_df_window, time_window_size, devices)
                    if random.random() < test_size:
                        test_samples.append(sample)
                    else:
                        train_samples.append(sample)
    print("Sample creation complete.")
    return train_samples, test_samples

def save_samples(samples, output_file):
    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        else:
            return obj

    # Recursively convert all elements in the samples list
    samples = json.loads(json.dumps(samples, default=convert_numpy_types))

    # create output dir if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving samples to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(samples, f)
    print("Samples saved.")

def get_most_recent_time_stamp_dir(data_dir):
    timestamps = os.listdir(os.path.join(data_dir, 'darshan_logs'))
    timestamps.sort()
    return timestamps[-1]

def get_time_stamp_dir(data_dir, time_stamp_str):
    timestamps = os.listdir(os.path.join(data_dir, 'darshan_logs'))
    for timestamp in timestamps:
        if time_stamp_str in timestamp:
            return timestamp
    raise ValueError(f"Time stamp directory {time_stamp_str} not found in {data_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_txt', action='store_true', help='Run the script with txt files instead of darshan files')
    args = parser.parse_args()
    print("Starting main process...")
    data_config = load_data_config()
    if data_config['time_stamp_dir'] == 'most_recent':
        data_config['time_stamp_dir'] = get_most_recent_time_stamp_dir(os.path.join(IOSENSE_ROOT, "data", data_config['workload']))
    else:
        data_config['time_stamp_dir'] = get_time_stamp_dir(os.path.join(IOSENSE_ROOT, "data", data_config['workload']), data_config['time_stamp_dir'])
    data_config['cluster_config'] = load_cluster_config()
    window_sizes = data_config['window_sizes']
    data, devices = get_data(data_config, args.run_txt)
    print(f"Devices: {devices}")
    for window_size in window_sizes:
        data_config['time_window_size'] = window_size
        print(f"Processing window size: {window_size}")
        train_samples, test_samples = create_samples(data, window_size, data_config['test_size'], devices)
        time_stamp_dir = data_config['time_stamp_dir']
        save_samples(train_samples, os.path.join(IOSENSE_ROOT, data_config['output_dir'], data_config['workload'], time_stamp_dir, f'train_samples_{window_size}.json'))
        save_samples(test_samples, os.path.join(IOSENSE_ROOT, data_config['output_dir'], data_config['workload'], time_stamp_dir, f'test_samples_{window_size}.json'))
    print("Main process complete.")

if __name__ == "__main__":
    main()
    