# -*- coding: utf-8 -*-
"""
將 CWRU 軸承數據集 (ABCD_Datasets.pickle) 轉換並導出為多頁籤 Excel 試算表 (CWRU_Bearing_Dataset.xlsx)
"""

import os
import pickle
import numpy as np
import pandas as pd

def export_to_excel():
    print("[Excel 導出] 正在讀取 ./data/ABCD_Datasets.pickle 數據集...")
    pickle_path = os.path.join(".", "data", "ABCD_Datasets.pickle")
    excel_output_path = os.path.join(".", "CWRU_Bearing_Dataset.xlsx")

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"找不到數據檔: {pickle_path}，請先執行 run_data_prep.py！")

    with open(pickle_path, "rb") as f:
        ABCD_Datasets = pickle.load(f)

    X_train_D = ABCD_Datasets["train_datasets"]["D"]
    Y_train_D = ABCD_Datasets["train_labels"]["D"]

    # 1. 建立 Sheet 1: 類別總覽表 (Dataset_Summary)
    class_info = [
        {"Class_ID": 1, "Class_Name": "Normal Baseline", "Fault_Location": "Normal (無故障)", "Fault_Size_Inch": 0.000, "Description": "正常軸承 baseline 震動訊號"},
        {"Class_ID": 2, "Class_Name": "0.007_Inner_Race", "Fault_Location": "Inner Race (內圈故障)", "Fault_Size_Inch": 0.007, "Description": "0.007 英吋微小內圈點蝕故障"},
        {"Class_ID": 3, "Class_Name": "0.007_Ball", "Fault_Location": "Ball (滾珠故障)", "Fault_Size_Inch": 0.007, "Description": "0.007 英吋微小滾珠點蝕故障"},
        {"Class_ID": 4, "Class_Name": "0.007_Outer_Race", "Fault_Location": "Outer Race (外圈故障)", "Fault_Size_Inch": 0.007, "Description": "0.007 英吋微小外圈點蝕故障"},
        {"Class_ID": 5, "Class_Name": "0.014_Inner_Race", "Fault_Location": "Inner Race (內圈故障)", "Fault_Size_Inch": 0.014, "Description": "0.014 英吋中度內圈點蝕故障"},
        {"Class_ID": 6, "Class_Name": "0.014_Ball", "Fault_Location": "Ball (滾珠故障)", "Fault_Size_Inch": 0.014, "Description": "0.014 英吋中度滾珠點蝕故障"},
        {"Class_ID": 7, "Class_Name": "0.014_Outer_Race", "Fault_Location": "Outer Race (外圈故障)", "Fault_Size_Inch": 0.014, "Description": "0.014 英吋中度外圈點蝕故障"},
        {"Class_ID": 8, "Class_Name": "0.021_Inner_Race", "Fault_Location": "Inner Race (內圈故障)", "Fault_Size_Inch": 0.021, "Description": "0.021 英吋重度內圈點蝕故障"},
        {"Class_ID": 9, "Class_Name": "0.021_Ball", "Fault_Location": "Ball (滾珠故障)", "Fault_Size_Inch": 0.021, "Description": "0.021 英吋重度滾珠點蝕故障"},
        {"Class_ID": 10, "Class_Name": "0.021_Outer_Race", "Fault_Location": "Outer Race (外圈故障)", "Fault_Size_Inch": 0.021, "Description": "0.021 英吋重度外圈點蝕故障"}
    ]
    df_summary = pd.DataFrame(class_info)

    # 2. 建立 Sheet 2: 10 類別時域震動點陣表 (Raw_Vibration_Waveforms - 2048 Points)
    waveforms_dict = {"TimeStep": np.arange(1, 2049)}
    for fault_type in range(1, 11):
        idxTF = (Y_train_D == fault_type).flatten()
        matching_indices = np.where(idxTF)[0]
        if len(matching_indices) > 0:
            idxNo = matching_indices[0]
            # Channel 0: DE (Drive End), Channel 1: FE (Fan End)
            waveforms_dict[f"Class_{fault_type}_DE"] = X_train_D[idxNo][:2048, 0]
            waveforms_dict[f"Class_{fault_type}_FE"] = X_train_D[idxNo][:2048, 1]

    df_waveforms = pd.DataFrame(waveforms_dict)

    # 3. 建立 Sheet 3: 抽樣樣本特徵統計表 (Sample_Feature_Statistics - 200 筆樣本)
    sample_stats = []
    num_samples_to_export = min(200, len(X_train_D))
    for i in range(num_samples_to_export):
        c_id = int(Y_train_D[i][0])
        c_name = class_info[c_id - 1]["Class_Name"]
        de_sig = X_train_D[i][:, 0]
        fe_sig = X_train_D[i][:, 1]

        sample_stats.append({
            "Sample_ID": i + 1,
            "Class_ID": c_id,
            "Class_Name": c_name,
            "DE_Mean": np.mean(de_sig),
            "DE_Std": np.std(de_sig),
            "DE_Max": np.max(de_sig),
            "DE_Min": np.min(de_sig),
            "FE_Mean": np.mean(fe_sig),
            "FE_Std": np.std(fe_sig),
            "FE_Max": np.max(fe_sig),
            "FE_Min": np.min(fe_sig),
        })
    df_sample_stats = pd.DataFrame(sample_stats)

    # 導出至多頁籤 Excel
    print(f"[Excel 導出] 正在寫入 Excel 試算表至 {excel_output_path} ...")
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='1_Dataset_Summary', index=False)
        df_waveforms.to_excel(writer, sheet_name='2_Vibration_Waveforms', index=False)
        df_sample_stats.to_excel(writer, sheet_name='3_Sample_Statistics', index=False)

    print(f"[Excel 導出] 成功生成 Excel 檔案: {excel_output_path} (大小: {os.path.getsize(excel_output_path)} bytes)")

if __name__ == '__main__':
    export_to_excel()
