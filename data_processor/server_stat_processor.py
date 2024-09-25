"""
    This Script processes stat files which were genrated during benchmark runs.
    The following statistics are gathered at each time step for each OST and MDT:

    
    read I/Os, write I/Os, discard I/0s
    ===================================

    These values increment when an I/O request completes.

    read merges, write merges, discard merges
    =========================================

    These values increment when an I/O request is merged with an
    already-queued I/O request.

    read sectors, write sectors, discard_sectors
    ============================================

    These values count the number of sectors read from, written to, or
    discarded from this block device.  The "sectors" in question are the
    standard UNIX 512-byte sectors, not any device- or filesystem-specific
    block size.  The counters are incremented when the I/O completes.

    read ticks, write ticks, discard ticks
    ======================================

    These values count the number of milliseconds that I/O requests have
    waited on this block device.  If there are multiple I/O requests waiting,
    these values will increase at a rate greater than 1000/second; for
    example, if 60 read requests wait for an average of 30 ms, the read_ticks
    field will increase by 60*30 = 1800.

    in_progress
    =========

    This value counts the number of I/O requests that have been issued to
    the device driver but have not yet completed.  It does not include I/O
    requests that are in the queue but not yet issued to the device driver.

    io_ticks
    ========

    This value counts the number of milliseconds during which the device has
    had I/O requests queued.

    time_in_queue
    =============

    This value counts the number of milliseconds that I/O requests have waited
    on this block device.  If there are multiple I/O requests waiting, this
    value will increase as the product of the number of milliseconds times the
    number of requests waiting (see "read ticks" above for an example).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

STAT_DIR = "/root/darshan-analysis/applications/dlio_bench/stats"


def get_stats_df(stat_folder, stat_dir=STAT_DIR):
    """
    Reads all stat files in stat_dir and returns a pandas dataframe.
    """
    stats_path = os.path.join(stat_dir, stat_folder)
    server_targets = None
    for config in os.listdir(stats_path):
        if config.endswith('.tgz'):
            continue
        sub_config_dir = os.listdir(os.path.join(stats_path, config))[0]
        stat_file_path = os.path.join(stats_path, config, sub_config_dir)
        server_files = [f for f in os.listdir(stat_file_path) if f.startswith('stats') and not f.endswith('.tar.gz')]
        column_order = ['time_stamp', 'read_ios', 'read_merges', 'sectors_read', 'time_reading', 
                        'write_ios', 'write_merges', 'sectors_written', 'time_writing', 
                        'in_progress', 'io_time', 'weighted_io_time', 'config']
        
        for server in server_files:
            print(f'Processing {server}')
            server_name = re.match(r'stats-(.*?)-.*', server).group(1)
            for idx, stat_file in enumerate(os.listdir(f'{stat_file_path}/{server}')):
                column_order = {column: [] for column in column_order}
                target = idx
                with open(f'{stat_file_path}/{server}/{stat_file}', 'r') as f:
                    #iterate by 2 lines at a time
                    lines = f.readlines()
                    previous_line = None
                    for i in range(0, len(lines), 2):
                        column_order['config'].append(config)
                        line = lines[i].strip()
                        # re match timestamp
                        timestamp_format = '%Y-%m-%d_%H-%M-%S'
                        time_stamp = datetime.strptime(line, timestamp_format)
                        if time_stamp == 0:
                            continue
                        column_order['time_stamp'].append(time_stamp.timestamp())
                        # re match stat line
                        line = lines[i+1]
                        stat_line = list(re.match(r'\s*?(\S+)\s*?(\S+)\s*?(\S+)\s*?(\S+)\s*?(\S+)\s*?(\S+)\s*?(\S+)\s*?(\S+)\s*?(\S+)\s*?(\S+)\s*?(.*)', line).groups())
                        for j, column in enumerate(list(column_order.keys())[1:-1]):
                            if i == 0:
                                column_order[column].append(0)
                            elif j == 8:
                                column_order[column].append(int(stat_line[j]))
                            else:
                                if previous_line is None:
                                    column_order[column].append(0)
                                else:
                                    if int(stat_line[j]) < int(previous_line[j]):
                                        # stat has reset at 32 bit max
                                        column_order[column].append(int(stat_line[j]) + (2**32 - int(previous_line[j])))
                                    else:
                                        column_order[column].append(int(stat_line[j])-int(previous_line[j]))
                        previous_line = stat_line
                for i, column in enumerate(column_order):
                    if i != 8:
                        column_order[column][0] = 0
                server_column = [server_name]*len(column_order['read_ios'])
                target_column = [target]*len(column_order['read_ios'])
                if server_targets is None:
                    server_targets = pd.DataFrame({**column_order, 'server': server_column, 'target': target_column})
                else:
                    server_targets = pd.concat([server_targets, pd.DataFrame({**column_order, 'server': server_column, 'target': target_column})])
    # drop rows where time_stamp is 0
    server_targets = server_targets[server_targets['time_stamp'] != 0]
    server_targets = server_targets.sort_values(by=['time_stamp'])
    server_targets = server_targets.reset_index()
    return server_targets

def plot_stats(server_targets, stat, server, target, plot_type='line'):
    """
    Plots the given stat for the given server and target.
    """
    stat_df = server_targets[(server_targets['server'] == server) & (server_targets['target'] == target)]
    fig, ax = plt.subplots()
    if plot_type == 'line':
        stat_df.plot.line(use_index=True, y='in_progress', ax=ax)
    elif plot_type == 'bar':
        stat_df.plot.bar(x='time_reading', y=stat)
    else:
        raise ValueError(f'Invalid plot type {plot_type}')
    plt.savefig(f'{stat}_{server}_{target}.png')


if __name__ == "__main__":
    folder = 'multi_config_run_2023-12-20_13-04'
    server_targets = get_stats_df(folder)
    print(server_targets)
    for server in server_targets['server'].unique():
        for stat in server_targets.columns[1:-3]:

            if stat == 'time_stamp':
                continue
            # print descritive stats
            print(f"{server}, {stat}: {server_targets[(server_targets['server'] == server)][stat].describe()}")
        #for target in server_targets['target'].unique():
        #print(f"{server}, {target}: {len(server_targets[(server_targets['server'] == server) & (server_targets['target'] == target)])}")
        #print(f"{server}, {target}: ")
    server_targets.to_csv(f"{os.path.join(STAT_DIR, 'csv_files', folder.split('/')[0])}.csv")
    plot_stats(server_targets, 'in_progress', 'slave1', '28069')
