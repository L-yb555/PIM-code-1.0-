import os


def list_csv_files(filepath):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            basename_splited = file.split(', ')
            if basename_splited[-1] != 'emg.csv':
                continue
                
            emg_filepath = os.path.join(root, file)
            basename_splited[-1] = 'pps.csv'
            pps_filepath = os.path.join(root, ', '.join(basename_splited))
            basename_splited[-1] = 'pose.csv'
            pose_filepath = os.path.join(root, ', '.join(basename_splited))
            basename_splited[-1] = 'merged.csv'
            merged_filepath = os.path.join(root, ', '.join(basename_splited))
            basename_splited[-1] = 'join.csv'
            join_filepath = os.path.join(root, ', '.join(basename_splited))
            paths.append((emg_filepath, pps_filepath, pose_filepath, merged_filepath, join_filepath))
    return paths


def list_merged_files(filepath):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            basename_splited = file.split(', ')
            if basename_splited[-1] != 'merged.csv':
                continue

            merged_filepath = os.path.join(root, file)
            basename_splited[-1] = 'figure.png'
            figure_filepath = os.path.join(root, ', '.join(basename_splited))

            paths.append((merged_filepath, figure_filepath))
    return paths
