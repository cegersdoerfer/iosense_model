import pandas as pd
import numpy as np



def parse_darshan_txt(txt_output, devices):
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
    osts = []
    mdt = []
    # Variables to hold temporary data
    current_file_id = None
    current_rank = None
    current_api = 'POSIX'
    trace_start_time = None

    mdt_width = len(devices['mdt'])
    
    ost_width = len(devices['ost'])

    # Parse the txt output
    for line in txt_output.splitlines():
        print(line)
        # Extract start time
        if line.startswith("# start_time:"):
            trace_start_time = float(line.split(':')[1].strip())
        
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
            # Check if the line has the expected number of fields
            if len(parts) < 8:
                continue
            current_api = parts[0]
            if not "POSIX" in current_api:
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
                ost_array = np.zeros(ost_width)
                mdt_array = np.zeros(mdt_width)
                if operation in ['read', 'write']:  
                    for ost in parts[9:]:
                        if ']' in ost:
                            ost = ost.replace(']', '')
                        ost_array[int(ost)] = 1
                else:
                    mdt_array[0] = 1
                osts.append(ost_array)
                mdt.append(mdt_array)

    osts = np.array(osts)
    mdt = np.array(mdt)
                
    # Create DataFrame
    dataframe_columns = {
        'file_id': file_ids,
        'api': apis,
        'rank': ranks,
        'operation': operations,
        'segment': segments,
        'offset': offsets,
        'size': sizes,
        'start': starts,
        'end': ends
    }
    for i in devices['ost']:
        int_id = int(i.replace('ost_', ''))
        dataframe_columns[f'ost_{int_id}'] = osts[:, int_id]
    

    # skip mdt for now and treat it as a single device
    #for i in devices['mdt']:
    #    int_id = int(i.replace('mdt_', ''))
    #    dataframe_columns[f'mdt_{int_id}'] = mdt[:, int_id]

    dataframe_columns['mdt_0'] = mdt[:, 0]
    for col in dataframe_columns:
        print(col, len(dataframe_columns[col]))
    df = pd.DataFrame(dataframe_columns)
    df = pd.DataFrame.from_dict(df).sort_values(by=['start'])
    df.reset_index(inplace=True)
    
    return df, trace_start_time, full_runtime
