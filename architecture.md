# MATLAB (.mat) 數據批量轉換與 TimeStep 標準欄位架構設計

## 1. 系統概述
本系統旨在將 `data/download` 目錄下的 30 個 `.mat` 格式軸承數據檔案，統一注入**採樣點序號 (`TimeStep`)** 與標籤屬性，使所有 30 個 `.xlsx` 導出文件具有**完全一致的欄位名稱、順序與數量（10 大標準欄位）**。`TimeStep` 採用純整數索引 (1, 2, 3, ...)，便利所有外部 AI 訓練與數據分析平台做時間序列步長定位。

## 2. 10 大標準欄位結構 (Standardized 10-Column Schema)
所有 30 個 `.xlsx` 檔案將統一規範為以下 10 個欄位（依序排列）：

1. **`Class_ID`** (int): 軸承診斷類別編號 (1 ~ 10)
2. **`Class_Name`** (str): 軸承診斷類別名稱 (如 `Normal Baseline`, `0.007_Inner_Race` 等)
3. **`Fault_Location`** (str): 故障位置 (`Normal`, `Inner Race`, `Ball`, `Outer Race`)
4. **`Fault_Size_Inch`** (float): 點蝕損壞尺寸 (0.000, 0.007, 0.014, 0.021)
5. **`Load_HP`** (int): 馬達運轉負載數值 (`0`, `1`, `2` 馬力)
6. **`TimeStep`** (int): **採樣點序號整數** ($1, 2, 3, \dots, N$，純整數時間步)
7. **`DE_time`** (float): 驅動端震動時域訊號 (Drive End Vibration Signal)
8. **`FE_time`** (float): 風扇端震動時域訊號 (Fan End Vibration Signal)
9. **`BA_time`** (float): 軸承座震動時域訊號 (Base Acceleration Signal，若原檔缺失則自動填入 `NaN`)
10. **`RPM`** (float/int): 馬達運轉轉速 (若原檔缺失則根據 CWRU 官方紀錄自動修補：0HP=1797, 1HP=1772, 2HP=1750)

## 3. TimeStep 純整數計算規範
- `TimeStep`: 從 `1` 開始整數遞增至數據總列數 $N$（如 1, 2, 3, ..., 121991），確保簡潔無小數浮點干擾。


## 4. 執行與並行加速
- **並行加速**：採用 8 進程 (`ProcessPoolExecutor`) 高效導出。
- **結構與型態強校驗**：寫入後即時檢測每個 XLSX 檔案欄位數是否精確等於 10，驗證 `TimeStep` 為正整數型態。

