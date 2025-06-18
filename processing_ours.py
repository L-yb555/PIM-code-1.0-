import numpy as np
import pandas as pd
import argparse
import os
from sklearn.preprocessing import StandardScaler

import pickle
from sklearn.preprocessing import PolynomialFeatures


parser = argparse.ArgumentParser(description='Posture-Informed Muscular Force Learning for Robust Hand Pressure Estimation')
FLAGS = parser.parse_args()

# Raw data info
emg_fps = 2000

file_ids = list(range(0, 22))
file_ids_train = list(range(0, 22))
file_ids_test = list(range(0, 22))

total_session = 3
session_ids = list(range(1, total_session))

# Data processing
start_time = 0.0
duration = 28 
window_length = 256 
hop_length = 32
# Dataset generation
num_frames = 32
hop_length_train = 4
hop_length_test = 8

def process_emg_and_force(args):
    valid_sessions = set(args.train_sessions + args.test_sessions)
    for sid in session_ids:
        if sid not in valid_sessions:  # <---- 只处理 train 和 test 的 session
            continue
        for fid in file_ids:
            # EMG data 8个通道 在join里面
            print(os.path.join(args.data_path, "Session{:d}".format(sid), "join_{:d}.csv".format(fid)))
            emg = pd.read_csv(os.path.join(args.data_path, "Session{:d}".format(sid), "join_{:d}.csv".format(fid)), dtype=np.float64, float_precision='high')
            emg = emg.iloc[:, 1:]

            emg_signal_names = [column for column in emg.columns if 'sEMG' in column]
            manus_signal_names = [column for column in emg.columns if '[' in column]

            sdscaler = StandardScaler()
            sdscaler.fit(emg[manus_signal_names])
            sdscaled_data = pd.DataFrame(sdscaler.transform(emg[manus_signal_names]))
            emg = pd.concat([emg.iloc[:,0], emg[emg_signal_names], sdscaled_data], axis=1)

            # remove outlier of emg
            emg_remove = np.array(emg[emg_signal_names])
            emg_remove = pd.DataFrame(emg_remove)
            emg = pd.concat([emg.iloc[:,0], emg_remove, sdscaled_data], axis=1)
            emg = np.array(emg.transpose())
            emg[0] = np.linspace(start=0, stop=emg[0][-1] - emg[0][0], num=len(emg[0]))
            emg = emg[:, emg[0] != 0.0] 
            begin = 0
            end = 0
            for timestamp in emg[0]:
                if timestamp < start_time:
                    begin += 1
                else:
                    break
            for timestamp in emg[0][::-1]:
                if timestamp > start_time + duration:
                    end -= 1
                else:
                    break

            emg = emg[1:, begin:end] if end < 0 else emg[1:, begin:]  

            num_frames_original = emg.shape[1] // hop_length - window_length // hop_length + 1
            fps = num_frames_original / float(duration)

            emg = emg.transpose()[:(num_frames_original + window_length // hop_length - 1) * hop_length]
            print("EMG data shape:", emg.shape)
            np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_{:d}.npy".format(fid)), emg)
            
            # Marbledex data fsr 5个指尖的压力
            force = pd.read_csv(os.path.join(args.data_path, "Session{:d}".format(sid), "fsr_{:d}.csv".format(fid)), dtype=np.float64, float_precision='high')
            force = force.drop_duplicates(['Timestamp'])
                                    
            marble_names = [column for column in force.columns if 'FSR' in column]
            marble = np.array(force[marble_names])

            force = np.concatenate([np.array(force.iloc[:,0]).reshape(-1,1), marble], 1)

            force = np.array(force.transpose())
            force[0] = np.linspace(start=0, stop=force[0][-1] - force[0][0], num=len(force[0]))
            force = np.array(force.transpose())

            # Resample force data
            timestamps = np.linspace(start_time + window_length / (2.0 * emg_fps), duration + start_time - window_length / (2.0 * emg_fps), num=num_frames_original, endpoint=True, dtype=np.float32)
            force_downsampled = force[np.abs(force[:, 0].reshape(1, -1) - timestamps.reshape(-1, 1)).argmin(1)]           
            force_downsampled[np.abs(force[:, 0].reshape(1, -1) - timestamps.reshape(-1, 1)).min(1) >= (1.0 / fps)] = 0 

            force = force_downsampled[:, 1:]

            # load the regression model
            predict_model = pickle.load(open('Checkpoints/regression_model.sav', 'rb'))

            poly_features  = PolynomialFeatures(degree=5, include_bias=False)
            force_newton = []
            for i in range(5):
                force_poly = poly_features.fit_transform(force[:, i].reshape(-1,1))
                force_newton.append(predict_model.predict(force_poly))
            
            force_newton = np.hstack(force_newton)*0.5

            # PPS data
            pps_force = pd.read_csv(
                os.path.join(args.data_path, "Session{:d}".format(sid), "pps_{:d}.csv".format(fid)),
                dtype=np.float64
            )
            pps_force = pps_force.drop_duplicates(['Time [ms]'])
            # 创建参考时间轴（保持为 float 秒）
            # 创建参考时间轴（保持为 float 秒）
            ref_time = np.linspace(
                start=pps_force.iloc[0, 0],
                stop=pps_force.iloc[-1, 0],
                num=2975
            )
            ref_time_df = pd.DataFrame({'Time [ms]': ref_time})
            # 不要转成 datetime，直接使用 float
            # pps_force['Time [ms]'] 也保持为 float
            # 使用 float 时间戳进行 merge_asof
            force_interpolation = pd.merge_asof(
                ref_time_df,
                pps_force,
                on='Time [ms]',  # 修复了引号闭合的问题
                direction='nearest',
                tolerance=0.01  # 单位是秒
            )

            # 插值非时间列
            for column in force_interpolation.columns:
                if column != 'Time [ms]':
                    force_interpolation[column] = force_interpolation[column].interpolate(method='linear')

            force_interpolation.iloc[:, 0] = force_interpolation.iloc[:, 0].values.astype("float64")
            pps_force = force_interpolation
                        
            palm_rightup = ['Elem15','Elem16']
            palm_leftup = ['Elem17','Elem18']
            palm_rightdown = ['Elem0','Elem1', 'Elem5','Elem6', 'Elem9','Elem10']
            palm_leftdown = ['Elem2','Elem3', 'Elem7','Elem8', 'Elem11','Elem12']

            # Definition of fingers and palm
            pps_force[pps_force < 0] = 0
            pps_force['V5'] = pps_force[palm_rightup].max(axis=1)
            pps_force['V7'] = pps_force[palm_leftup].max(axis=1)
            pps_force['V6'] = np.mean([pps_force[palm_rightdown].max(1),np.sort(pps_force[palm_rightdown], axis=1)[:, -2]], axis=0)
            pps_force['V8'] = np.mean([pps_force[palm_leftdown].max(1),np.sort(pps_force[palm_leftdown], axis=1)[:, -2]], axis=0)

            pps_names = [column for column in pps_force.columns if 'V' in column]

            force_pps = np.array(pps_force[pps_names])
            
            force_pps[force_pps < 0] = 0
            
            force_pps = np.concatenate([np.array(pps_force.iloc[:,0]).reshape(-1,1), force_pps], 1)

            force_pps = np.array(force_pps.transpose())
            force_pps[0] = np.linspace(start=0, stop=30, num=len(force_pps[0])) 
            
            force_pps = np.array(force_pps.transpose())
            
            # Resample force data
            timestamps = np.linspace(start_time + window_length / (2.0 * emg_fps), duration + start_time - window_length / (2.0 * emg_fps), num=num_frames_original, endpoint=True, dtype=np.float32)
            pps_downsampled = force_pps[np.abs(force_pps[:, 0].reshape(1, -1) - timestamps.reshape(-1, 1)).argmin(1)]

            force_pps = pps_downsampled[:, 1:] / 2 
            force_pps[force_pps < 0.8] = 0
            force_pps = force_pps * 2
            force_pps = force_pps * 0.689009

            force_newton = np.concatenate([force_newton, force_pps],1)

            force_newton[force_newton < 0.2] = 0
            force_newton[force_newton > 20] = 20
            
            force_class = np.asarray(force_newton >= 1, dtype=np.int64)

            print("Force data shape:", force_newton.shape)
            scale = 8 # 250
            np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "force_{:d}.npy".format(fid)), force_newton / scale)
            np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "force_class_{:d}.npy".format(fid)), force_class)

def build_dataset(args):
    emg_train = []
    force_train = []
    force_class_train = []
    for sid in session_ids:
        for fid in file_ids_train:
            emg = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_{:d}.npy".format(fid)))
            force = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "force_{:d}.npy".format(fid)))
            force_class = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "force_class_{:d}.npy".format(fid)))
            if sid in args.train_sessions:
                for cid in range(0, force.shape[0] - num_frames + 1, hop_length_train):
                    emg_train.append(emg[cid * hop_length:cid * hop_length + (num_frames + window_length // hop_length - 1) * hop_length])
                    force_train.append(force[cid:cid + num_frames])
                    force_class_train.append(force_class[cid:cid + num_frames])
                print("Training file {:d} from session {:d} done!".format(fid, sid))
    emg_train = np.stack(emg_train, axis=0).transpose(0, 2, 1)
    force_train = np.stack(force_train, axis=0).transpose(0, 2, 1)
    force_class_train = np.stack(force_class_train, axis=0).transpose(0, 2, 1)
    print("Training EMG data shape:", emg_train.shape)
    print("Training force shape:", force_train.shape)
    print("Training force class shape:", force_class_train.shape)
    np.save(os.path.join(args.dataset_path, "emg_train"), emg_train)
    np.save(os.path.join(args.dataset_path, "force_train"), force_train)
    np.save(os.path.join(args.dataset_path, "force_class_train"), force_class_train)

    for sid in session_ids:
        for fid in file_ids_test:
            emg = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_{:d}.npy".format(fid)))
            force = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "force_{:d}.npy".format(fid)))
            force_class = np.load(os.path.join(args.data_path, "Session{:d}".format(sid), "force_class_{:d}.npy".format(fid)))

            if sid in args.train_sessions:
                pass
            elif sid in args.test_sessions:
                emg_test = []
                force_test = []
                force_class_test = []
                for cid in range(0, force.shape[0] - num_frames + 1, hop_length_test):
                    emg_test.append(emg[cid * hop_length:cid * hop_length + (num_frames + window_length // hop_length - 1) * hop_length])
                    force_test.append(force[cid:cid + num_frames])
                    force_class_test.append(force_class[cid:cid + num_frames])
                emg_test = np.stack(emg_test, axis=0).transpose(0, 2, 1)
                force_test = np.stack(force_test, axis=0).transpose(0, 2, 1)
                force_class_test = np.stack(force_class_test, axis=0).transpose(0, 2, 1)


                np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "emg_test_{:d}.npy".format(fid)), emg_test)
                np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "force_test_{:d}.npy".format(fid)), force_test)
                np.save(os.path.join(args.data_path, "Session{:d}".format(sid), "force_class_test_{:d}.npy".format(fid)), force_class_test)
                print("Evaluation file {:d} from session {:d} done!".format(fid, sid))

def main(args):
    print('cw:', os.getcwd())
    args.data_path = os.path.join("Dataset/")
    print(args.data_path)
    args.dataset_path = os.path.join(os.getcwd(), 'Dataset')
    if not os.path.exists(args.data_path):
        raise Exception("Data not found!")
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    args.train_sessions = list(range(1, total_session, 3))
    args.test_sessions = list(range(2, total_session, 3))

    process_emg_and_force(args)
    build_dataset(args)


if __name__ == '__main__':
    main(FLAGS)
