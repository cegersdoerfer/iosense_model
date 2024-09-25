import json
import os
import pandas as pd
import subprocess
from data_processor.parse_darshan_txt import parse_darshan_txt
import re
import argparse


IOSENSE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_model_config():
    with open(os.path.join(IOSENSE_ROOT,'iosense_model','model_config.json'), 'r') as f:
        return json.load(f)

def load_cluster_config(path):
    print(f"Loading cluster configuration from {path}...")
    with open(path, 'r') as f:
        config = json.load(f)
    print("Cluster configuration loaded.")
    return config

def load_darshan_trace_from_dir(dir_path, config_name, run_txt):
    print(f"Loading Darshan traces from {dir_path} for config {config_name}...")
    traces = []
    for file in os.listdir(dir_path):
        if not run_txt:
            if file.endswith('.darshan') and config_name in file:
                print(f"Processing file: {file}")
                darshan_txt = subprocess.check_output(["darshan-dxt-parser", "--show-incomplete", os.path.join(dir_path, file)])
                darshan_txt = darshan_txt.decode('utf-8')
                trace_df, trace_start_time, trace_runtime = parse_darshan_txt(darshan_txt)
                traces.append(trace_df)
        else:
            if file.endswith('.txt') and config_name in file:
                print(f"Processing file: {file}")
                with open(os.path.join(dir_path, file), 'r') as f:
                    trace_txt = f.read()
                trace_df, trace_start_time, trace_runtime = parse_darshan_txt(trace_txt)
                traces.append(trace_df)
    if traces:
        darshan_df = pd.concat(traces)
        darshan_df = darshan_df.sort_values(by=['start'])
        darshan_df.reset_index(inplace=True)
        darshan_df = darshan_df.drop(columns=['index'])
        print("Darshan traces loaded and concatenated.")
        return darshan_df
    else:
        print("No Darshan traces found.")
        return pd.DataFrame()

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
            for column_idx, column in enumerate(columns[4:]):
                if line_idx == 0:
                    first_line = line
                    data[column].append(line[column_idx])
                else:
                    # for in_flight, do not calculate difference
                    if column == 'in_flight':
                        data[column].append(int(line[column_idx]))
                    else:
                        if int(line[column_idx]) < int(data[column][line_idx - 1]):
                            data[column].append(int(line[column_idx]) + (2**32 - int(data[column][line_idx - 1])))
                        else:
                            data[column].append(int(line[column_idx]) - int(data[column][line_idx - 1]))
        for column_idx, column in enumerate(columns[4:]):
            data[column][0] = int(first_line[column_idx])
    df = pd.DataFrame(data)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.sort_values(by=['time_stamp'], inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    print(f"Stats file {file_path} processed.")
    return df

def load_stats_from_dir(dir_path):
    print(f"Loading stats from directory: {dir_path}")
    stats = {}
    for folder in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, folder)):
            continue
        if 'stats' not in folder:
            continue
        else:
            subdirs = os.listdir(os.path.join(dir_path, folder))
            for subdir in subdirs:
                if any([x in subdir for x in ['OST', 'MDT']]):
                    # dirs will be in fsname_[MDT|OST][xxxx].log, like hasanfs-MDT0000.log, extract MDT or OST and the number in xxxx
                    match = re.match(r'.*?-(MDT[0-9]+|OST[0-9]+)\.log', subdir)
                    if match:
                        id = match.group(1)
                        print(f"Processing subdir: {subdir}, ID: {id}")
                        if id not in stats:
                            stats[id] = []
                        stats[id].append(process_stats_file(os.path.join(dir_path, folder, subdir)))
    for id in stats:
        stats[id] = pd.concat(stats[id])
        stats[id]['time_stamp'] = pd.to_datetime(stats[id]['time_stamp'], format='%Y-%m-%d %H:%M:%S.%f')
        stats[id].sort_values(by=['time_stamp'], inplace=True)
        stats[id].reset_index(inplace=True)
    print("Stats loaded and concatenated.")
    return stats
            


def get_data(model_config, run_txt):
    print("Getting data based on model configuration...")
    data = {'baseline_traces': {}, 'interference_traces': {}}
    data_types = ['darshan_logs', 'stats']
    workload = model_config['workload']
    time_stamp_dir = model_config['time_stamp_dir']
    data_dir = model_config['cluster_config']['data_dir']
    for data_type in data_types:
        if data_type == 'darshan_logs':
            interference_levels = os.listdir(os.path.join(data_dir, workload, data_type, time_stamp_dir))
            for interference_level in interference_levels:
                interference_level_num = int(interference_level.split('_')[-1])
                interference_level_files = os.listdir(os.path.join(data_dir, workload, data_type, time_stamp_dir, interference_level))
                configs = []
                for file in interference_level_files:
                    # file will be in format configname_idx.darshan like debug_config_0.darshan
                    print(f"Processing file: {file}")
                    config_name = re.match(r'(.*?)_[0-9].darshan', file).group(1)
                    if config_name not in configs:
                        configs.append(config_name)
                for config in configs:
                    trace_df = load_darshan_trace_from_dir(os.path.join(data_dir, workload, data_type, time_stamp_dir, interference_level), config, run_txt)
                    if interference_level_num == 0:
                        data['baseline_traces'][config] = trace_df
                    else:
                        if interference_level_num not in data['interference_traces']:
                            data['interference_traces'][interference_level_num] = {}
                        data['interference_traces'][interference_level_num][config] = trace_df
        elif data_type == 'stats':
            data['stats'] = load_stats_from_dir(os.path.join(data_dir, workload, data_type, time_stamp_dir))
    print("Data retrieval complete.")
    return data



def get_trace_features(trace_df_window):
    window_runtime = trace_df_window['end'].max() - trace_df_window['start'].min()

    trace_features = {}
    trace_features['num_ops'] = len(trace_df_window)
    trace_features['num_read_ops'] = len(trace_df_window[trace_df_window['api'] == 'read'])
    trace_features['num_write_ops'] = len(trace_df_window[trace_df_window['api'] == 'write'])
    if window_runtime > 0:
        trace_features['num_read_ops_per_sec'] = trace_features['num_read_ops'] / window_runtime
        trace_features['num_write_ops_per_sec'] = trace_features['num_write_ops'] / window_runtime
    else:
        trace_features['num_read_ops_per_sec'] = 0
        trace_features['num_write_ops_per_sec'] = 0

    trace_features['size_read_ops'] = trace_df_window[trace_df_window['operation'] == 'read']['size'].sum()
    trace_features['size_write_ops'] = trace_df_window[trace_df_window['operation'] == 'write']['size'].sum()
    if window_runtime > 0:
        trace_features['size_read_ops_per_sec'] = trace_features['size_read_ops'] / window_runtime
        trace_features['size_write_ops_per_sec'] = trace_features['size_write_ops'] / window_runtime
    else:
        trace_features['size_read_ops_per_sec'] = 0
        trace_features['size_write_ops_per_sec'] = 0
    return trace_features, window_runtime

def get_stats_features(stats_df_window, time_window_size):
    stats_features = {}
    for device in stats_df_window:
        total_read_ios = stats_df_window[device]['read_ios'].sum()
        total_write_ios = stats_df_window[device]['write_ios'].sum()
        sectors_read = stats_df_window[device]['sectors_read'].sum()
        sectors_written = stats_df_window[device]['sectors_written'].sum()
        read_ticks = stats_df_window[device]['read_ticks'].sum()
        write_ticks = stats_df_window[device]['write_ticks'].sum()
        time_in_queue = stats_df_window[device]['time_in_queue'].sum()

        stats_features[f'{device}_read_throughput'] = sectors_read / time_window_size
        stats_features[f'{device}_write_throughput'] = sectors_written / time_window_size
        stats_features[f'{device}_read_iops'] = total_read_ios / time_window_size
        stats_features[f'{device}_write_iops'] = total_write_ios / time_window_size
        stats_features[f'{device}_read_ticks'] = read_ticks
        stats_features[f'{device}_write_ticks'] = write_ticks
        stats_features[f'{device}_time_in_queue'] = time_in_queue
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

    

def calculate_sample(trace_df_window, baseline_trace_df_window, stats_df_window, time_window_size):
    trace_features, window_runtime = get_trace_features(trace_df_window)
    if window_runtime > 0:
        time_window_size = window_runtime
    print(f"Time window size: {time_window_size}")
    stats_features = get_stats_features(stats_df_window, time_window_size)
    absolute_runtime_diff, relative_runtime_diff = get_label(baseline_trace_df_window, window_runtime)
    sample = {
        'trace_features': trace_features,
        'stats_features': stats_features,
        'absolute_runtime_diff': absolute_runtime_diff,
        'relative_runtime_diff': relative_runtime_diff
    }
    return sample

def create_samples(data, time_window_size):
    print(f"Creating samples with time window size: {time_window_size}")
    samples = []
    # minimum size is 0.2 seconds
    if time_window_size < 0.2:
        # throw error
        raise ValueError('Time window size must be at least 0.2 seconds')
    
    for interference_level in data['interference_traces']:
        for config in data['interference_traces'][interference_level]:
            trace_df = data['interference_traces'][interference_level][config]
            baseline_trace_df = data['baseline_traces'][config]
            trace_start_time = trace_df['start'].min()
            trace_end_time = trace_df['end'].max()
            num_windows = int((trace_end_time - trace_start_time) / time_window_size)
            for i in range(num_windows):
                start_time = trace_start_time + i * time_window_size
                end_time = trace_start_time + (i + 1) * time_window_size
                trace_df_window = trace_df[(trace_df['start'] >= start_time) & (trace_df['start'] < end_time)]
                indices = trace_df_window['index'].tolist()
                # get the same operation indices in the baseline trace
                baseline_trace_df_window = baseline_trace_df.iloc[indices]
                # get the stats for the same time window
                stats_df_window = {}
                for device in data['stats']:
                    stats_df_window[device] = data['stats'][device][(data['stats'][device]['time_stamp'] >= start_time) & (data['stats'][device]['time_stamp'] < end_time)]
                sample = calculate_sample(trace_df_window, baseline_trace_df_window, stats_df_window, time_window_size)
                samples.append(sample)
    print("Sample creation complete.")
    return samples

def save_samples(samples, output_file):

    # create output dir if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Saving samples to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(samples, f)
    print("Samples saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_txt', action='store_true', help='Run the script with txt files instead of darshan files')
    args = parser.parse_args()
    print("Starting main process...")
    model_config = load_model_config()
    cluster_config = load_cluster_config(os.path.join(IOSENSE_ROOT, 'client', model_config['cluster_config']))
    model_config['cluster_config'] = cluster_config
    print("Model and cluster configurations loaded.")
    data = get_data(model_config, args.run_txt)
    samples = create_samples(data, 0.2)
    save_samples(samples, os.path.join(IOSENSE_ROOT, model_config['output_dir'], model_config['workload'], model_config['time_stamp_dir'], 'all_samples.json'))
    print("Main process complete.")

if __name__ == "__main__":
    main()
    