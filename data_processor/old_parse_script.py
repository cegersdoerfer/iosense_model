import pandas as pd
import numpy as np



def parse_darshan_txt(txt_output):
    # Lists to store extracted data
    file_ids = []
    apis = []
    ranks = []
    operations = []
    segments = []
    offsets = []
    sizes = []
    starts = []
    ends = []
    servers = []
    # Variables to hold temporary data
    current_file_id = None
    current_rank = None
    current_api = 'POSIX'
    trace_start_time = None
    
    for line in txt_output.splitlines():
        # Extract start time
        if line.startswith("# start_time:"):
            trace_start_time = float(line.split(':')[1].strip())
            trace_start_time -= 3600
        
        if line.startswith("# run time:"):
            full_runtime = float(line.split(':')[1].strip())

        # Extract file_id
        if line.startswith("# DXT, file_id:"):
            current_file_id = line.split(':')[1].split(',')[0].strip()
            
        # Extract rank
        if line.startswith("# DXT, rank:"):
            current_rank = line.split(':')[1].split(',')[0].strip()
            
        # Extract IO operation details
        if not line.startswith("#") and current_file_id and current_rank:
            parts = line.split()
            # each line has the following fields: # Module    Rank  Operation  Segment          Offset       Length    Start(s)      End(s)  [OST]
            # Check if the line has the expected number of fields
            if len(parts) < 8:
                continue
            operation = parts[2]
            operations.append(operation)
            ranks.append(current_rank)
            file_ids.append(current_file_id)
            apis.append(current_api)
            segments.append(int(parts[3]))
            if parts[4] == 'N/A':
                offsets.append(0)
            else:
                offsets.append(int(parts[4]))
            if parts[5] == 'N/A':
                sizes.append(0)
            else:
                sizes.append(int(parts[5])/1000000)
            starts.append(float(parts[6]) + trace_start_time)
            ends.append(float(parts[7]) + trace_start_time)
            if len(parts) >= 9:
                cur_servers = np.zeros(12)
                for ost in parts[9:]:
                    if ']' in ost:
                        ost = ost.replace(']', '')
                    if ost == 'mdt':
                        cur_servers[6] = 1
                    else:
                        cur_servers[int(ost)] = 1
                servers.append(cur_servers)
            else:
                pass
                #servers.append(np.zeros(7))
    #servers = np.array(servers)
                
    # Create DataFrame
    df = pd.DataFrame({
        'file_id': file_ids,
        'api': apis,
        'rank': ranks,
        'operation': operations,
        'segment': segments,
        'offset': offsets,
        'size': sizes,
        'start': starts,
        'end': ends,
        #'ost0': servers[:, 0],
        #'ost1': servers[:, 1],
        #'ost2': servers[:, 2],
        #'ost3': servers[:, 3],
        #'ost4': servers[:, 4],
        #'ost5': servers[:, 5],
        #'mdt': servers[:, 6]
    })
    df = pd.DataFrame.from_dict(df).sort_values(by=['start'])
    df.reset_index(inplace=True)
    
    return df, trace_start_time, full_runtime