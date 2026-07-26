# -*- coding: utf-8 -*-
"""
1D-CNN 軸承故障智能預測/推理腳本 (Bearing Fault Prediction & Inference Script)
輸入 2048 點震動訊號切片，自動輸出故障診斷類別、置信度與維護建議。
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_model():
    CNNch = 2
    modelC2 = Sequential()
    modelC2.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same', activation='relu', input_shape=(2048, CNNch)))
    modelC2.add(MaxPooling1D(pool_size=2))
    modelC2.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu'))
    modelC2.add(MaxPooling1D(pool_size=2))
    modelC2.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    modelC2.add(MaxPooling1D(pool_size=2))
    modelC2.add(Dropout(0.2))
    modelC2.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    modelC2.add(MaxPooling1D(pool_size=2))
    modelC2.add(Dropout(0.2))
    modelC2.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    modelC2.add(MaxPooling1D(pool_size=2))
    modelC2.add(Flatten())
    modelC2.add(Dense(50, activation='relu'))
    modelC2.add(Dropout(0.2))
    modelC2.add(Dense(10, activation='softmax'))
    return modelC2

class_info = [
    {"Class_ID": 1, "Name": "Normal Baseline", "Location": "無故障 (Normal)", "Severity": "0.000 (正常健康)", "Recommendation": "軸承狀態良好，請繼續保持定期監測與正常潤滑。"},
    {"Class_ID": 2, "Name": "0.007_Inner_Race", "Location": "內圈故障 (Inner Race)", "Severity": "0.007 英吋 (輕微點蝕)", "Recommendation": "檢測到微小內圈點蝕，建議安排於下次例行維護時複查。"},
    {"Class_ID": 3, "Name": "0.007_Ball", "Location": "滾珠故障 (Ball)", "Severity": "0.007 英吋 (輕微點蝕)", "Recommendation": "檢測到微小滾珠點蝕，請補充潤滑油脂並密切關注振幅變化。"},
    {"Class_ID": 4, "Name": "0.007_Outer_Race", "Location": "外圈故障 (Outer Race)", "Severity": "0.007 英吋 (輕微點蝕)", "Recommendation": "檢測到微小外圈點蝕，建議持續記錄高頻震動趨勢。"},
    {"Class_ID": 5, "Name": "0.014_Inner_Race", "Location": "內圈故障 (Inner Race)", "Severity": "0.014 英吋 (中度損壞)", "Recommendation": "檢測到中度內圈損傷，建議準備備用軸承零件並排定替換計劃。"},
    {"Class_ID": 6, "Name": "0.014_Ball", "Location": "滾珠故障 (Ball)", "Severity": "0.014 英吋 (中度損壞)", "Recommendation": "檢測到中度滾珠損傷，請注意軸承運轉溫度，防範卡死風險。"},
    {"Class_ID": 7, "Name": "0.014_Outer_Race", "Location": "外圈故障 (Outer Race)", "Severity": "0.014 英吋 (中度損壞)", "Recommendation": "檢測到中度外圈損傷，請檢查機台安裝基座與對心狀況。"},
    {"Class_ID": 8, "Name": "0.021_Inner_Race", "Location": "內圈故障 (Inner Race)", "Severity": "0.021 英吋 (重度剝落)", "Recommendation": "【警告】檢測到嚴重內圈剝落，請儘速安排計劃性停機更換軸承！"},
    {"Class_ID": 9, "Name": "0.021_Ball", "Location": "滾珠故障 (Ball)", "Severity": "0.021 英吋 (重度破損)", "Recommendation": "【警告】檢測到嚴重滾珠破損，極易引發機台劇烈震動，建議立即檢檢修！"},
    {"Class_ID": 10, "Name": "0.021_Outer_Race", "Location": "外圈故障 (Outer Race)", "Severity": "0.021 英吋 (重度裂化)", "Recommendation": "【警告】檢測到嚴重外圈裂化剝落，請立即切換備用機並進行關機檢修！"}
]

def predict_vibration_sample(sample_signal, true_label=None):
    """
    輸入單筆 (2048, 2) 的震動訊號，進行故障預測並輸出報告
    """
    weights_path = 'CNNC2.weights.best.weights.h5'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到模型權重檔 {weights_path}，請先執行 run_1DCNN_demo.py 訓練模型！")

    model = build_model()
    model.load_weights(weights_path)

    if len(sample_signal.shape) == 2:
        sample_input = np.expand_dims(sample_signal, axis=0)
    else:
        sample_input = sample_signal

    preds = model.predict(sample_input, verbose=0)[0]
    pred_class_idx = int(np.argmax(preds))
    confidence = float(preds[pred_class_idx]) * 100
    info = class_info[pred_class_idx]

    print("\n" + "="*65)
    print("           1D-CNN 軸承故障智能預測診斷報告 (Diagnostic Report)")
    print("="*65)
    if true_label is not None:
        print(f"  【真實標籤 (True Label)】   : Class {true_label}")
    print(f"  【預測類別 (Predicted Class)】: Class {info['Class_ID']} ({info['Name']})")
    print(f"  【診斷置信度 (Confidence)】   : {confidence:.2f}%")
    print(f"  【故障部位 (Location)】      : {info['Location']}")
    print(f"  【損傷程度 (Severity)】      : {info['Severity']}")
    print(f"  【工廠維護建議 (Action)】    : {info['Recommendation']}")
    print("="*65 + "\n")

    return info, confidence

def main():
    print("[智能預測] 讀取 ./data/ABCD_Datasets.pickle 隨機測試樣本進行推理測試...")
    pickle_path = os.path.join(".", "data", "ABCD_Datasets.pickle")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"找不到數據檔: {pickle_path}！")

    with open(pickle_path, "rb") as f:
        ABCD_Datasets = pickle.load(f)

    X_test_D = ABCD_Datasets["test_datasets"]["D"]
    Y_test_D = ABCD_Datasets["test_labels"]["D"]

    # 隨機抽取 3 筆測試集樣本進行預測展示
    random_indices = np.random.choice(len(X_test_D), size=3, replace=False)
    for idx in random_indices:
        sample_signal = X_test_D[idx]
        true_label = int(Y_test_D[idx][0])
        predict_vibration_sample(sample_signal, true_label=true_label)

if __name__ == '__main__':
    main()
