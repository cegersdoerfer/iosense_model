import pandas as pd


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
    osts = []
    # Variables to hold temporary data
    current_file_id = None
    current_rank = None
    current_api = 'POSIX'
    trace_start_time = None
    
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
            parts = line.split()
            # Check if the line has the expected number of fields
            if len(parts) < 8:
                continue
            operations.append(parts[2])
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
                ost_info = ','.join(parts[9:]).replace(']', '')
                osts.append(ost_info)
            else:
                osts.append('')
                
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
        'ost': osts
    })
    df = pd.DataFrame.from_dict(df).sort_values(by=['start'])
    df.reset_index(inplace=True)
    
    return df, trace_start_time, full_runtime
