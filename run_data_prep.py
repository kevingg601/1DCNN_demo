# -*- coding: utf-8 -*-
"""
數據預處理腳本: 下載 CWRU Bearing 數據集並將數據打包為 ABCD_Datasets.pickle 格式。
"""

import os
import sys
import urllib.request
import scipy.io
import numpy as np
import pickle

url_base = 'https://engineering.case.edu/sites/default/files/'
download_folder = os.path.join('.', 'data', 'download')

if not os.path.exists(download_folder):
    os.makedirs(download_folder, exist_ok=True)

file_dictionary = {
    "1-A": "98.mat",
    "1-B": "99.mat",
    "1-C": "100.mat",
    "2-A": "119.mat",
    "2-B": "120.mat",
    "2-C": "121.mat",
    "3-A": "186.mat",
    "3-B": "187.mat",
    "3-C": "188.mat",
    "4-A": "223.mat",
    "4-B": "224.mat",
    "4-C": "225.mat",
    "5-A": "106.mat",
    "5-B": "107.mat",
    "5-C": "108.mat",
    "6-A": "170.mat",
    "6-B": "171.mat",
    "6-C": "172.mat",
    "7-A": "210.mat",
    "7-B": "211.mat",
    "7-C": "212.mat",
    "8-A": "131.mat",
    "8-B": "132.mat",
    "8-C": "133.mat",
    "9-A": "198.mat",
    "9-B": "199.mat",
    "9-C": "200.mat",
    "10-A": "235.mat",
    "10-B": "236.mat",
    "10-C": "237.mat"
}

import ssl

def download_files():
    print("[數據預處理] 開始檢查並下載 .mat 數據檔...")
    req_headers = {'User-Agent': 'Mozilla/5.0'}
    ssl_context = ssl._create_unverified_context()
    for key, filename in file_dictionary.items():
        dest_filename = os.path.join(download_folder, filename)
        if not os.path.exists(dest_filename) or os.path.getsize(dest_filename) == 0:
            print(f"正在下載: {filename} ...")
            url = url_base + filename
            try:
                req = urllib.request.Request(url, headers=req_headers)
                with urllib.request.urlopen(req, context=ssl_context) as response, open(dest_filename, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"下載完成: {filename} ({os.path.getsize(dest_filename)} bytes)")
            except Exception as e:
                print(f"下載失敗 {filename}: {e}")
                raise e
        else:
            print(f"檔案已存在: {filename}")

def prepare_pickle():
    print("[數據預處理] 開始讀取 .mat 檔並進行特徵擷取...")
    Channels = ['DE', 'FE']
    Vib_mc = {}
    for c in Channels:
        Sen_series = {}
        for key, filename in sorted(file_dictionary.items()):
            mtfilename = os.path.join(download_folder, filename)
            data = scipy.io.loadmat(mtfilename)
            for i in data:
                if '_' + c + '_time' in i:
                    Sen_series[key] = data[i]
        Vib_mc[c] = Sen_series

    sample_len = 2048
    sample_test_cnt = 25

    X_test_temp = {}
    for key in sorted(Vib_mc[Channels[0]].keys()):
        X_test_temp[key] = np.concatenate((
            Vib_mc[Channels[0]][key][:sample_len*sample_test_cnt].reshape(sample_test_cnt, sample_len, 1),
            Vib_mc[Channels[1]][key][:sample_len*sample_test_cnt].reshape(sample_test_cnt, sample_len, 1)
        ), axis=2)

    X_train_temp = {}
    sample_train_cnt = 660
    sample_train_shift = 103

    for key in sorted(Vib_mc[Channels[0]].keys()):
        train_array = np.concatenate((
            Vib_mc[Channels[0]][key][sample_len*sample_test_cnt:sample_len*sample_test_cnt+sample_len].reshape(1, sample_len, 1),
            Vib_mc[Channels[1]][key][sample_len*sample_test_cnt:sample_len*sample_test_cnt+sample_len].reshape(1, sample_len, 1)
        ), axis=2)
        for i in range(sample_train_cnt - 1):
            train_array = np.concatenate((
                train_array,
                np.concatenate((
                    Vib_mc[Channels[0]][key][sample_len*sample_test_cnt+sample_train_shift*(i+1):sample_len*sample_test_cnt+sample_len+sample_train_shift*(i+1)].reshape(1, sample_len, 1),
                    Vib_mc[Channels[1]][key][sample_len*sample_test_cnt+sample_train_shift*(i+1):sample_len*sample_test_cnt+sample_len+sample_train_shift*(i+1)].reshape(1, sample_len, 1)
                ), axis=2)
            ), axis=0)
        X_train_temp[key] = train_array

    def randomize(dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation, :]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def create_dataset(X_temp):
        X = {"A": np.empty((0, sample_len, 2), float), "B": np.empty((0, sample_len, 2), float), "C": np.empty((0, sample_len, 2), float)}
        Y = {"A": np.empty((0, 1), int), "B": np.empty((0, 1), int), "C": np.empty((0, 1), int)}
        for key, sample in sorted(X_temp.items()):
            hp_key = key.split(sep="-")[1]
            cat_key = int(key.split(sep="-")[0])
            X[hp_key] = np.concatenate((X[hp_key], sample), axis=0)
            y = np.empty(len(sample), int)
            y.fill(cat_key)
            y = y.reshape((len(sample), 1))
            Y[hp_key] = np.concatenate((Y[hp_key], y), axis=0)
        for key, samplearray in sorted(X.items()):
            X[key], Y[key] = randomize(samplearray, Y[key])
        return X, Y

    X_test, Y_test = create_dataset(X_test_temp)
    X_train, Y_train = create_dataset(X_train_temp)

    X_test["D"], Y_test["D"] = randomize(
        np.vstack((X_test["A"], X_test["B"], X_test["C"])),
        np.vstack((Y_test["A"], Y_test["B"], Y_test["C"]))
    )

    X_train["D"], Y_train["D"] = randomize(
        np.vstack((X_train["A"], X_train["B"], X_train["C"])),
        np.vstack((Y_train["A"], Y_train["B"], Y_train["C"]))
    )

    pickle_file = os.path.join('.', 'data', 'ABCD_Datasets.pickle')
    with open(pickle_file, 'wb') as f:
        save = {
            'train_datasets': X_train,
            'train_labels': Y_train,
            'test_datasets': X_test,
            'test_labels': Y_test,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

    print(f"[數據預處理] 數據集打包完成: {pickle_file} (大小: {os.path.getsize(pickle_file)} bytes)")

if __name__ == '__main__':
    download_files()
    prepare_pickle()
