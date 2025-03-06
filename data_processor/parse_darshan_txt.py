import pandas as pd
import numpy as np



def parse_darshan_txt(txt_output, devices, file_ids_offsets_osts_map=None):
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
    num_lines = 0
    for line in txt_output.splitlines():
        
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
            num_lines += 1
            parts = line.split()
            # Check if the line has the expected number of fields
            if len(parts) < 9:
                if len(parts) < 8:
                    continue
                operation = parts[2]
                offset_start = int(parts[4])
                offset_end = offset_start + int(parts[5])
                size = int(parts[5])/1000000
                
                if current_file_id in file_ids_offsets_osts_map:
                    # check if offset and size are within any of the offsets and sizes in the file_ids_offsets_osts_map[current_file_id]
                    #print(f"current_file_id: {current_file_id}")
                    ost_arrays = []
                    mdt_arrays = []
                    for offset_tuple in file_ids_offsets_osts_map[current_file_id]:
                        if offset_start >= offset_tuple[0] and offset_end <= offset_tuple[1]:
                            ost_arrays.append(file_ids_offsets_osts_map[current_file_id][offset_tuple]["ost"])
                            mdt_arrays.append(file_ids_offsets_osts_map[current_file_id][offset_tuple]["mdt"])
                            break
                        elif offset_start >= offset_tuple[0] and offset_end >= offset_tuple[1]:
                            ost_arrays.append(file_ids_offsets_osts_map[current_file_id][offset_tuple]["ost"])
                            mdt_arrays.append(file_ids_offsets_osts_map[current_file_id][offset_tuple]["mdt"])
                            offset_start = offset_tuple[1] + 1

                    operations.append(operation)
                    ranks.append(current_rank)
                    file_ids.append(current_file_id)
                    apis.append(current_api)
                    segments.append(int(parts[3]))
                    offsets.append(offset_start)
                    sizes.append(size)
                    starts.append(float(parts[6]) + trace_start_time)
                    ends.append(float(parts[7]) + trace_start_time)
                    osts_arrays = np.array(ost_arrays)
                    mdt_arrays = np.array(mdt_arrays)
                    summed_osts = np.sum(osts_arrays, axis=0)
                    summed_mdt = np.sum(mdt_arrays, axis=0)
                    summed_osts[summed_osts > 0] = 1
                    summed_mdt[summed_mdt > 0] = 1
                    summed_osts = summed_osts.astype(int).flatten()
                    summed_mdt = summed_mdt.astype(int).flatten()
                    osts.append(summed_osts)
                    mdt.append(summed_mdt)

                else:
                    pass
                    #print(f"id_tuple not found: {id_tuple}")
                continue
            else:
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
                if current_file_id not in file_ids_offsets_osts_map:
                    file_ids_offsets_osts_map[current_file_id] = {}
                offset_start = int(parts[4])
                offset_end = offset_start + int(parts[5])
                offset_tuple = (offset_start, offset_end)
                file_ids_offsets_osts_map[current_file_id][offset_tuple] = {"ost": ost_array, "mdt": mdt_array}

    print(f"num_lines: {num_lines}, len(operations): {len(operations)}")
    if num_lines > 100 and len(operations) == 0:
        return None, None, None, file_ids_offsets_osts_map, True

    if len(osts) == 0:
        return None, trace_start_time, full_runtime, file_ids_offsets_osts_map, True
    
    osts = np.array(osts)
    mdt = np.array(mdt)
    

    print(osts.shape)
    print(mdt.shape)
                
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
    
    return df, trace_start_time, full_runtime, file_ids_offsets_osts_map, False
