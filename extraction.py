import os
import yaml
from datetime import datetime, timedelta

# Load configuration from config.yaml (assumes it’s in the same directory as extraction.py)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Now set extraction parameters from the config
PREICTAL_DURATION = config.get("PREICTAL_DURATION", 3600)   # 60 minutes
PREICTAL_OFFSET   = config.get("PREICTAL_OFFSET", 300)        # 5 minutes before seizure onset
INVALID_DURATION  = config.get("INVALID_DURATION", 14400)     # 4 hours

# ----- Helper Functions -----

def parse_time(time_str):
    """
    Parse a time string (e.g., '27:09:34') into a datetime object.
    """
    hours, minutes, seconds = map(int, time_str.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    base_date = datetime(2000, 1, 1)
    return base_date + timedelta(seconds=total_seconds)

def parse_summary_file(source):
    """
    Read the summary file and return a list of file dictionaries.
    Accepts either a file path (str) or a file-like object with a read() method.
    Each dictionary contains:
      - 'filename'
      - 'start_time' and 'end_time' as datetime objects (adjusted for day rollovers)
      - 'seizures': list of dictionaries with keys 'start' and 'end' (seconds relative to file start)
    
    If a file’s parsed start_time is earlier than the previous file’s end_time,
    whole days (24-hour blocks) are added until the order is maintained.
    """
    # Determine if source is a path or a file-like object
    if isinstance(source, str):
        # Open the file path for reading.
        with open(source, 'r') as f:  # <-- Changed summary_source to source here
            lines = f.readlines()
    elif hasattr(source, "read"):
        # It is already a file-like object.
        lines = source.readlines()
    else:
        raise ValueError("source must be a file path or a file-like object with a read() method.")

    files_info = []
    current_file = None
    prev_end = None  # holds the adjusted end_time of the previous file

    for line in lines:
        line = line.strip()
        if line.startswith('File Name:'):
            if current_file:
                files_info.append(current_file)
            current_file = {'filename': line.split(': ')[1].strip(), 'seizures': []}
        elif line.startswith('File Start Time:'):
            t = parse_time(line.split(': ')[1].strip())
            if prev_end is not None:
                while t < prev_end:
                    t += timedelta(days=1)
            current_file['start_time'] = t
        elif line.startswith('File End Time:'):
            t = parse_time(line.split(': ')[1].strip())
            while t < current_file['start_time']:
                t += timedelta(days=1)
            current_file['end_time'] = t
            prev_end = t
        elif 'Seizure' in line and 'Start Time:' in line:
            seizure_start = float(line.split(': ')[1].strip().split(' ')[0])
            current_file['seizures'].append({'start': seizure_start, 'end': None})
        elif 'Seizure' in line and 'End Time:' in line:
            seizure_end = float(line.split(': ')[1].strip().split(' ')[0])
            if current_file['seizures']:
                current_file['seizures'][-1]['end'] = seizure_end

    if current_file:
        files_info.append(current_file)

    return files_info



def compute_global_file_times(files_info):
    """
    For each file, compute:
      - global_start: seconds from the earliest file's start_time
      - duration: (end_time - start_time).total_seconds()
      - global_end: global_start + duration
    """
    earliest = min(f['start_time'] for f in files_info)
    for f in files_info:
        f['global_start'] = (f['start_time'] - earliest).total_seconds()
        f['duration'] = (f['end_time'] - f['start_time']).total_seconds()
        f['global_end'] = f['global_start'] + f['duration']
    return files_info

def extract_global_seizures(files_info):
    """
    Convert each seizure’s start and end (relative to file start) to global times.
    Returns a sorted list of seizure dictionaries with keys 'start' and 'end'.
    """
    global_seizures = []
    for f in files_info:
        for sz in f['seizures']:
            global_seizures.append({
                'start': f['global_start'] + sz['start'],
                'end': f['global_start'] + sz['end']
            })
    global_seizures.sort(key=lambda s: s['start'])
    return global_seizures

def merge_seizure_clusters(seizures, merge_threshold=INVALID_DURATION):
    """
    Merge seizures into clusters if the gap between one seizure's end and the next seizure's start is ≤ merge_threshold.
    Returns a list of clusters (each a list of seizure dictionaries).
    """
    if not seizures:
        return []
    clusters = []
    current_cluster = [seizures[0]]
    for sz in seizures[1:]:
        if sz['start'] - current_cluster[-1]['end'] <= merge_threshold:
            current_cluster.append(sz)
        else:
            clusters.append(current_cluster)
            current_cluster = [sz]
    if current_cluster:
        clusters.append(current_cluster)
    return clusters

def compute_global_intervals(global_seizures, global_end):
    """
    Compute global intervals for preictal and interictal data.
    
    For each seizure cluster:
      - Preictal: from max(previous invalid end, first_seizure.start - PREICTAL_DURATION, 0)
                  to first_seizure.start - PREICTAL_OFFSET.
      - Invalid (discard): from first_seizure.end to last_seizure.end + INVALID_DURATION.
    
    Interictal intervals are the gaps between the end of an invalid period and the start of the next preictal window,
    with any remaining time after the last invalid period also marked as interictal.
    """
    clusters = merge_seizure_clusters(global_seizures, merge_threshold=INVALID_DURATION)
    global_preictals = []
    interictals = []
    last_invalid_end = 0  # end of previous invalid period

    for cluster in clusters:
        first_seizure = cluster[0]
        last_seizure = cluster[-1]
        cand_pre_start = first_seizure['start'] - PREICTAL_DURATION
        cand_pre_end   = first_seizure['start'] - PREICTAL_OFFSET
        effective_pre_start = max(last_invalid_end, cand_pre_start, 0)
        if cand_pre_end > effective_pre_start:
            global_preictals.append((effective_pre_start, cand_pre_end))
        # Define invalid period for the cluster.
        invalid_interval = (first_seizure['end'], last_seizure['end'] + INVALID_DURATION)
        if last_invalid_end < effective_pre_start:
            interictals.append((last_invalid_end, effective_pre_start))
        last_invalid_end = invalid_interval[1]
    if last_invalid_end < global_end:
        interictals.append((last_invalid_end, global_end))
    
    return global_preictals, interictals

def map_intervals_to_file(file_info, intervals):
    """
    Given a file (with 'global_start' and 'global_end') and a list of global intervals (tuples),
    return a list of intervals (in file-relative seconds) corresponding to the intersection.
    """
    mapped = []
    file_start = file_info['global_start']
    file_end = file_info['global_end']
    for (g_start, g_end) in intervals:
        inter_start = max(file_start, g_start)
        inter_end = min(file_end, g_end)
        if inter_end > inter_start:
            mapped.append((inter_start - file_start, inter_end - file_start))
    mapped.sort(key=lambda x: x[0])
    return mapped

def extract_interictal_preictal(summary_path):
    """
    Main extraction function.
    Takes a single argument: the full path to the summary file.
    Returns a dictionary mapping each file to its preictal and interictal intervals.
    """
    files_info = parse_summary_file(summary_path)
    files_info = compute_global_file_times(files_info)
    global_end = max(f['global_end'] for f in files_info)
    global_seizures = extract_global_seizures(files_info)
    global_preictals, global_interictals = compute_global_intervals(global_seizures, global_end)
    
    extraction_dict = {}
    for f in files_info:
        extraction_dict[f['filename']] = {
            "Preictal": map_intervals_to_file(f, global_preictals),
            "Interictal": map_intervals_to_file(f, global_interictals)
        }
    return extraction_dict

# ----- Main Execution -----

def main(summary_path):
    extraction_dict = extract_interictal_preictal(summary_path)
    print("Final Extraction Dictionary:")
    for fname, ranges in extraction_dict.items():
        print(f"{fname}: {ranges}")
    return extraction_dict

if __name__ == "__main__":
    # Loop over each subject summary file (adjust the path as needed)
    for i in range(1, 24):
        summary_file = f"/Volumes/episense/chb-mit-scalp-eeg-database-1.0.0/chb{i:02d}/chb{i:02d}-summary.txt"
        main(summary_file)
