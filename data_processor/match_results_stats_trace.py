from server_stat_processor import get_stats_df
from parse_darshan_txt import parse_darshan_txt
import matplotlib.pyplot as plt
import argparse
import darshan
import pandas as pd
import os
import numpy as np
import datetime
import json
import subprocess

WORKLOAD = 'io500'
DLIO_MODEL = 'unet3d'
OST_SERVERS = ['slave0', 'slave1', 'slave2']
MDT_SERVERS = ['lab2']

PLOTS_DIR = f'/root/darshan-analysis/python-files/plots/{WORKLOAD}'


def get_darshan_results():
    # Get the results from the darshan traces
    path = os.path.join(TRACE_DIR, MULTI_CONFIG_RUN)
    darshan_traces = []
    for config in os.listdir(path):
        # iterate through all .darshan files
        config_name = config.split('_')[3]
        config_path = os.path.join(path, config)
        interference_level = config.split('_')[4]
        full_runtime = 0
        local_traces = []
        for idx, file in enumerate(os.listdir(config_path)):
            print(f'\t\t{file}')
            if file.endswith('.darshan'):
                
                darshan_txt = subprocess.check_output(["darshan-dxt-parser", "--show-incomplete", os.path.join(config_path, file)])
                darshan_txt = darshan_txt.decode('utf-8')
                print(f'\t\t{len(darshan_txt.splitlines())} lines found')
                trace_df, trace_start_time, trace_runtime = parse_darshan_txt(darshan_txt)
                full_runtime += trace_runtime
                local_traces.append(trace_df)
        for trace in local_traces:
            trace['full_runtime'] = full_runtime
            trace['config'] = config_name
            trace['interference_level'] = interference_level
            darshan_traces.append(trace)

    print(f'\t\t{len(darshan_traces)} traces found')
    darshan_traces = pd.concat(darshan_traces)
    darshan_traces = darshan_traces.sort_values(by=['start'])
    darshan_traces = darshan_traces.reset_index()
    darshan_traces = darshan_traces.drop(columns=['index'])
    print(f'\t\t{len(darshan_traces)} operations found')
    return darshan_traces


def get_server_stats():
    # Get the server stats
    server_stats = get_stats_df(MULTI_CONFIG_RUN, STATS_DIR)
    server_stats = server_stats.drop(columns=['index'])
    return server_stats

def plot_darshan_vs_time(darshan_traces, add_model_decisions=True):
    # Plot the IOPS vs time for the darshan traces
    # create a figure for each interference level and each interference config
    # 3 levels of interference and 2 configs
    fig_iops, ax_iops = plt.subplots(3, 2, figsize=(20, 20))
    fig_throughput, ax_throughput = plt.subplots(3, 2, figsize=(20, 20))
    for c_idx, config in enumerate(darshan_traces['config'].unique()):
        for i_idx, interference in enumerate(darshan_traces['interference_level'].unique()):
            trace = darshan_traces[(darshan_traces['config'] == config) & (darshan_traces['interference_level'] == interference)]
            # calculate the IOPS for each second based on start and end times
            trace_start = trace['start'].min()
            trace_end = trace['end'].max()
            trace_iops = []
            trace_throughput = []
            for i in range(int(trace_start), int(trace_end), 5):
                trace_iops.append(len(trace[(trace['start'] >= i) & (trace['end'] <= i+5)]))
                trace_throughput.append(trace[(trace['start'] >= i) & (trace['end'] <= i+5)]['size'].sum() / (1024 * 1024 * 5))
            ax_iops[i_idx, c_idx].set_title(f'{config} - {interference}')
            ax_iops[i_idx, c_idx].set_xlabel('Time (s)')
            ax_iops[i_idx, c_idx].set_ylabel('IOPS')
            # plot the average IOPS for each 5 second window
            ax_iops[i_idx, c_idx].plot(range(0, len(trace_iops)), trace_iops, label=f'{config} - {interference}', linewidth=0.5)
            # add a line to represent the average IOPS when IOPS value is not 0
            ax_iops[i_idx, c_idx].axhline(y=np.mean(trace_iops), color='r', linestyle='-', label=f'{config} - {interference} - mean')
            # multiply x axis by 5 to get the time in seconds and show only every 500 seconds
            time_ticks = [i for i in range(0, len(trace_iops), 500)]
            time_tick_labels = [str(i*5) for i in range(0, len(trace_iops), 500)]
            ax_iops[i_idx, c_idx].set_xticks(time_ticks)
            ax_iops[i_idx, c_idx].set_xticklabels(time_tick_labels)

            
            ax_throughput[i_idx, c_idx].set_title(f'{config} - {interference}')
            ax_throughput[i_idx, c_idx].set_xlabel('Time (s)')
            ax_throughput[i_idx, c_idx].set_ylabel('Throughput (MB/s)')
            ax_throughput[i_idx, c_idx].plot(range(0, len(trace_throughput)), trace_throughput, label=f'{config} - {interference}', linewidth=0.5)
            # add a line to represent the average throughput
            ax_throughput[i_idx, c_idx].axhline(y=np.mean(trace_throughput), color='r', linestyle='-', label=f'{config} - {interference} - mean')
            # multiply x axis by 5 to get the time in seconds and show only every 500 seconds
            ax_throughput[i_idx, c_idx].set_xticks(time_ticks)
            ax_throughput[i_idx, c_idx].set_xticklabels(time_tick_labels)


    if not os.path.isdir(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    # save the figures
    fig_iops.savefig(f'{PLOTS_DIR}/darshan_iops.png')
    fig_throughput.savefig(f'{PLOTS_DIR}/darshan_throughput.png')


def reset_indices(results_df, trace_df, stats_df):
    results_df = results_df.reset_index()
    results_df = results_df.drop(columns=['index'])

    trace_df = trace_df.reset_index()
    trace_df = trace_df.drop(columns=['index'])
    stats_df = stats_df.reset_index()
    stats_df = stats_df.drop(columns=['index'])

    return results_df, trace_df, stats_df

def match_benchmark_to_trace_and_stats(results_df, trace_df, stats_df):
    results_df, trace_df, stats_df = reset_indices(results_df, trace_df, stats_df)

    matched_dict = {}
    # Match the benchmark results to the darshan traces and server stats
    for i, row in results_df.iterrows():
        config = str(row['config'])
        section = str(row['section'])
        run = int(row['run'])
        if config not in matched_dict:
            matched_dict[config] = {}
        if section not in matched_dict[row['config']]:
            matched_dict[config][section] = {}
        matched_dict[config][section][run] = {'trace': None, 'ost_stats': None, 'mdt_stats': None, 'results': {'t_delta': row['t_delta']}}
        trace_match = trace_df[(trace_df['start'] >= row['t_start']) & (trace_df['end'] <= row['t_end'])]
        if len(trace_match) > 0:
            matched_dict[config][section][run]['trace'] = trace_match
        ost_stats_match = stats_df[(stats_df['time_stamp'] >= row['t_start']) & (stats_df['time_stamp'] <= row['t_end']) & (stats_df['server'].isin(OST_SERVERS))]
        if len(ost_stats_match) > 0:
            matched_dict[config][section][run]['ost_stats'] = ost_stats_match
        mdt_stats_match = stats_df[(stats_df['time_stamp'] >= row['t_start']) & (stats_df['time_stamp'] <= row['t_end']) & (stats_df['server'].isin(MDT_SERVERS))]
        if len(mdt_stats_match) > 0:
            matched_dict[config][section][run]['mdt_stats'] = mdt_stats_match
    return matched_dict

def make_op_windows(trace_df, window_size):
    # Make windows of operations
    windows = {}
    idx = 0
    for i in range(0, len(trace_df), window_size):
        windows[idx] = trace_df.iloc[i:i+window_size]
        idx += 1
    # Make sure the last window is the same size as the others
    if len(windows[list(windows.keys())[-1]]) != len(windows[0]):
        windows[list(windows.keys())[-1]] = trace_df.iloc[-len(windows[0]):]
    return windows

def make_time_windows(trace_df, window_size, baseline_trace, sliding_step_size):
    # Make windows of time
    windows = {}
    start = trace_df['start'].min()
    end = trace_df['end'].max()
    trace_df = trace_df.reset_index()
    baseline_trace = baseline_trace.drop(columns=['level_0'])
    baseline_trace = baseline_trace.reset_index()
    idx = 0
    for i in range(int(start), int(end), sliding_step_size):
        if len(trace_df[(trace_df['start'] >= i) & (trace_df['end'] <= i+window_size)]) == 0:
            idx += 1
            continue
        windows[idx] = {'trace': trace_df[(trace_df['start'] >= i) & (trace_df['end'] <= i+window_size)], 'i': i}
        baseline_match = baseline_trace.iloc[windows[idx]['trace'].index[0]:windows[idx]['trace'].index[-1]+1]
        windows[idx]['baseline'] = baseline_match['end'].max() - baseline_match['start'].min()
        if windows[idx]['baseline'] == 0:
            windows[idx]['baseline'] = baseline_match['full_runtime'].max()
        idx += 1
    return windows

def match_stats_to_trace(stats_df, trace_df, window_size, baseline_trace, time, sliding_step_size):
    # Match the server stats to the darshan trace
    windows = make_time_windows(trace_df, window_size, baseline_trace, sliding_step_size)
    print(f'\t\t{len(windows)} windows found')
    matched_stats = {}
    for window in windows:
        cur_window = windows[window]['trace']
        baseline = windows[window]['baseline']
        start = cur_window['start'].min()
        end = cur_window['end'].max()
        ost_stats_match = stats_df[(stats_df['time_stamp'] >= start) & (stats_df['time_stamp'] <= end)]
        mdt_stats_match = stats_df[(stats_df['time_stamp'] >= start) & (stats_df['time_stamp'] <= end)]
        matched_stats[window] = {'trace': cur_window, 'ost_stats': ost_stats_match, 'mdt_stats': mdt_stats_match}
        if time:
            matched_stats[window]['baseline'] = baseline
    return matched_stats

def get_multi_config_windows(stats_df, trace_df, window_size, sliding_step_size, baseline='None', time=False):
    # split the trace by config
    configs = trace_df['config'].unique()
    config_windows = {}
    baseline_trace = trace_df[trace_df['config'] == baseline]
    for config in configs:
        config_windows[config] = {}
        config_df = trace_df[trace_df['config'] == config]
        config_windows[config] = match_stats_to_trace(stats_df, config_df, window_size, baseline_trace, time, sliding_step_size)
    return config_windows

def summarize_stats(stats):
    summary = {}
    # iterate through servers
    for server in stats['server'].unique():
        server_stats = stats[stats['server'] == server]
        for column in server_stats.columns:
            if column not in ['time_stamp', 'config', 'server', 'target']:
                summary[f'{server}_{column}_mean'] = float(server_stats[column].mean())
                summary[f'{server}_{column}_std'] = float(server_stats[column].std())
                summary[f'{server}_{column}_sum'] = float(server_stats[column].sum())
    return summary

def summarize_trace(trace, window_size=None):
    summary = {}
    if window_size is not None:
        summary['total_time'] = float(trace['end'].max() - trace['start'].min())
        summary['window_size'] = float(trace['end'].max() - trace['start'].min())
    else:
        summary['total_time'] = float(trace['end'].max() - trace['start'].min())
        summary['window_size'] = float(trace['end'].max() - trace['start'].min())
    summary['total_ops'] = float(len(trace))
    summary['total_size'] = float(trace['size'].sum())
    summary['total_reads'] = float(len(trace[trace['operation'] == 'read']))
    summary['total_writes'] = float(len(trace[trace['operation'] == 'write']))
    summary['total_read_size'] = float(trace[trace['operation'] == 'read']['size'].sum())
    summary['total_write_size'] = float(trace[trace['operation'] == 'write']['size'].sum())
    summary['total_stat'] = float(len(trace[trace['operation'] == 'stat']))
    summary['total_open'] = float(len(trace[trace['operation'] == 'open']))
    if summary['total_time'] == 0:
        summary['IOPS'] = 0
        summary['read_IOPS'] = 0
        summary['write_IOPS'] = 0
        summary['throughput'] = 0
        summary['read_throughput'] = 0
        summary['write_throughput'] = 0
    else:
        summary['IOPS'] = float(summary['total_ops'] / summary['total_time'])
        summary['read_IOPS'] = float(summary['total_reads'] / summary['total_time'])
        summary['write_IOPS'] = float(summary['total_writes'] / summary['total_time'])
        summary['throughput'] = float(summary['total_size'] / summary['total_time'])
        summary['read_throughput'] = float(summary['total_read_size'] / summary['total_time'])
        summary['write_throughput'] = float(summary['total_write_size'] / summary['total_time'])
    return summary

def summarize_results(matched_dict, time=False, window_size=None):
    # Summarize the results
    threshold_exceeded = []
    summary = {}
    for config in matched_dict:
        summary[config] = {}
        for window in matched_dict[config]:
            window = int(window)
            summary[config][window] = {'trace': None, 'ost_stats': None, 'mdt_stats': None}
            if matched_dict[config][window]['trace'] is not None:
                trace = matched_dict[config][window]['trace']
                if time:
                    trace_summary = summarize_trace(trace, window_size=window_size)
                else:
                    trace_summary = summarize_trace(trace)
                if time:
                    if trace['end'].max() - trace['start'].min() == 0:
                        trace_summary['total_time'] = float((trace['end'].max() - trace['start'].min()) / matched_dict[config][window]['baseline'])
                    else:
                        trace_summary['total_time'] = float((trace['end'].max() - trace['start'].min()) / matched_dict[config][window]['baseline'])
                    if trace_summary['total_time'] > 5.0:
                        threshold_exceeded.append(trace_summary['total_time'])
                    else:
                        threshold_exceeded.append(0)
                summary[config][window]['trace'] = trace_summary
            if matched_dict[config][window]['ost_stats'] is not None:
                stats = matched_dict[config][window]['ost_stats']
                stats_summary = summarize_stats(stats)
                summary[config][window]['ost_stats'] = stats_summary
            if matched_dict[config][window]['mdt_stats'] is not None:
                stats = matched_dict[config][window]['mdt_stats']
                stats_summary = summarize_stats(stats)
                summary[config][window]['mdt_stats'] = stats_summary
    return summary, threshold_exceeded

def split_summary(summary, split=0.8):
    # Split the summary into train and test sets
    train_summary = {}
    test_summary = {}
    for config in summary:
        train_summary[config] = {}
        test_summary[config] = {}
        for window in summary[config]:
            choice = np.random.choice([True, False], p=[split, 1-split])
            if choice:
                train_summary[config][window] = summary[config][window]
            else:
                test_summary[config][window] = summary[config][window]
    return train_summary, test_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match the results of the benchmark to the darshan traces and server stats')
    parser.add_argument('--workload', type=str, default=WORKLOAD, help='The workload to analyze')
    parser.add_argument('--dlio_model', type=str, default=DLIO_MODEL, help='The dlio model to analyze')
    parser.add_argument('--multi_config_run', type=str, default='', help='The multi config run to analyze')
    parser.add_argument('--sliding_step_size', type=int, default=2, help='The sliding interval step size to use')

    args = parser.parse_args()

    MULTI_CONFIG_RUN = args.multi_config_run

    if args.workload == 'io500':
        STATS_DIR = '/root/darshan-analysis/applications/IO500/stats'
        TRACE_DIR = '/root/darshan-analysis/applications/IO500/darshan-traces'
    else:
        TRACE_DIR = f'/root/darshan-analysis/applications/{args.workload}/darshan-traces'
        STATS_DIR = f'/root/darshan-analysis/applications/{args.workload}/stats'
        TRACE_DIR = f'/root/darshan-analysis/applications/{args.workload}/darshan-traces'

    if args.workload == 'dlio_bench':
        DATA_DIR = f'/root/darshan-analysis/python-files/data_files/{args.workload}_{args.dlio_model}'
    else:
        DATA_DIR = f'/root/darshan-analysis/python-files/data_files/{args.workload}'

    PLOTS_DIR = f'/root/darshan-analysis/python-files/plots/{args.workload}'
    if args.workload == 'dlio_bench':
        PLOTS_DIR = f'/root/darshan-analysis/python-files/plots/{args.workload}_{args.dlio_model}'

    darshan_traces = get_darshan_results()
    #plot_darshan_vs_time(darshan_traces)
    server_stats = get_server_stats()
    time_window_list = np.arange(2, 5, 1)
    time_window_list = np.append(time_window_list, np.arange(5, 25, 5))
    for time_window_size in time_window_list:
        matched_dict = get_multi_config_windows(server_stats, darshan_traces, time_window_size, args.sliding_step_size, time=True)
        print(f'\t\t{len(matched_dict)} configs found')
        summary, threshold_exceeded = summarize_results(matched_dict, time=True, window_size=time_window_size)


        # plot threshold exceeded
        fig, ax = plt.subplots()
        ax.scatter(range(0, len(threshold_exceeded)), threshold_exceeded)
        ax.set_title(f'Threshold Exceeded - {args.workload} - {time_window_size}')
        ax.set_xlabel('Window')
        ax.set_ylabel('Threshold Exceeded')
        fig.savefig(f'{PLOTS_DIR}/{args.workload}_{time_window_size}_threshold_exceeded.png')


        train_summary, test_summary = split_summary(summary, split=0.8)
        print(f'\t\t{len(summary)} configs found')
        if not os.path.isdir(f'{DATA_DIR}/train'):
            os.mkdir(f'{DATA_DIR}/train')
        if not os.path.isdir(f'{DATA_DIR}/test'):
            os.mkdir(f'{DATA_DIR}/test')
        with open(f'{DATA_DIR}/train/{args.workload}_{time_window_size}_time_data.json', 'w') as f:
            json.dump(train_summary, f, indent=4)
        with open(f'{DATA_DIR}/test/{args.workload}_{time_window_size}_time_data.json', 'w') as f:
            json.dump(test_summary, f, indent=4)





    
    


