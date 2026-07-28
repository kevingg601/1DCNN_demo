# 1D-CNN 軸承故障智能診斷系統 (1D-CNN Bearing Fault Diagnosis System)

本專案基於 **1D-CNN (一維卷積神經網路)**，針對美國 **CWRU (Case Western Reserve University, 凱斯西儲大學)** 軸承數據集，實現端到端的震動訊號特徵學習與 **10 種軸承健康/故障狀態之精準分類診斷**。

> 📌 **致謝與原項目出處 (Original Repository)**：  
> 本專案改編與升級自原作者 **`meitetsu3`** 之 GitHub 專案：[https://github.com/meitetsu3/1DCNN](https://github.com/meitetsu3/1DCNN)。  
> 本版本全面升級了 **TensorFlow 2.x / Keras 3 語法相容**、補全**全繁體中文代碼與 Markdown 註解**、擴充 **10 種時域波形可視化**、**訓練曲線與混淆矩陣繪製**、**`ipywidgets` 互動預測控制面板**、**Excel 數據集導出**以及**全自動數據下載腳本**。

---

## 1. 數據集來源與自動下載指南 (Datasets & Download Guide)

### 數據集來源 (Data Origin)
* **原專案項目 (Original Project)**: [https://github.com/meitetsu3/1DCNN](https://github.com/meitetsu3/1DCNN)
* **官方數據中心**: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
* **官方檔案下載伺服器**: `https://engineering.case.edu/sites/default/files/`
* **採樣規格**: 加速度計採樣頻率為 12 kHz / 48 kHz，收錄電動馬達在 0 HP, 1 HP, 2 HP, 3 HP 四種負載下運轉之時域震動數據。
* **10 種診斷類別**:
  1. `Normal Baseline`（正常健康軸承）
  2. `0.007" Inner Race`（0.007 英吋微小內圈點蝕故障）
  3. `0.007" Ball`（0.007 英吋微小滾珠點蝕故障）
  4. `0.007" Outer Race`（0.007 英吋微小外圈點蝕故障）
  5. `0.014" Inner Race`（0.014 英吋中度內圈點蝕故障）
  6. `0.014" Ball`（0.014 英吋中度滾珠點蝕故障）
  7. `0.014" Outer Race`（0.014 英吋中度外圈點蝕故障）
  8. `0.021" Inner Race`（0.021 英吋重度內圈點蝕故障）
  9. `0.021" Ball`（0.021 英吋重度滾珠點蝕故障）
  10. `0.021" Outer Race`（0.021 英吋重度外圈點蝕故障）

### 全自動數據下載指令 (Automatic Download Command)
專案內建全自動下載與預處理腳本 `run_data_prep.py`，無須手動下載 30 個 `.mat` 檔案：

```powershell
# 執行自動下載與預處理（自動抓取 30 個 CWRU .mat 檔案並打包）
python run_data_prep.py
```
* 執行後會在 `./data/` 目錄下自動建立全套 `.mat` 檔與打包後的 `./data/ABCD_Datasets.pickle` (1.34 GB)。

> **註**：專案已配置 `.gitignore`，大型 `.mat` 原始檔、`.pickle` 數據集以及訓練權重 `*.h5` 均已自動排除於 Git 追蹤之外。

---

## 2. 專案主要檔案架構 (Project Structure)

| 檔案名稱 (File) | 說明 (Description) |
| :--- | :--- |
| **`1DCNN_demo.ipynb`** | **主要演示筆記本**。包含全繁體中文註解、TensorFlow 2.x 相容碼、原始震動波形圖、Loss/Acc 訓練歷史曲線、10x10 混淆矩陣，以及 `ipywidgets` 互動預測控制面板。 |
| **`run_data_prep.py`** | **自動數據準備腳本**。處理 SSL 驗證與 CWRU 官方網址變更，全自動下載 30 個 `.mat` 並封裝為 `ABCD_Datasets.pickle`。 |
| **`run_1DCNN_demo.py`** | **模型訓練與繪圖腳本**。訓練 5 層 1D-CNN 並自動導出 `1DCNN_training_history.png` 與 `1DCNN_confusion_matrix.png`。 |
| **`predict.py`** | **智能診斷推理腳本**。輸入 2,048 點震動訊號，自動輸出繁體中文故障診斷報告、置信度與維護建議。 |
| **`export_to_excel.py`** | **Excel 導出腳本**。將數據集轉換導出為 `CWRU_Bearing_Dataset.xlsx` 多頁籤試算表。 |
| **`.gitignore`** | **Git 排除清單**。自動忽略大型數據集、圖片與二進位檔。 |

---

## 3. 快速開始與使用步驟 (Quick Start)

### 步驟一：準備數據
```powershell
python run_data_prep.py
```

### 步驟二：訓練模型與產生圖表
```powershell
python run_1DCNN_demo.py
```

### 步驟三：執行智能預測與診斷
```powershell
python predict.py
```

### 步驟四：開啟互動式 Jupyter Notebook
```powershell
jupyter notebook 1DCNN_demo.ipynb
```

---

## 4. 參考來源與文獻 (References & Acknowledgements)

[1] 原作者 GitHub 專案: [meitetsu3/1DCNN](https://github.com/meitetsu3/1DCNN)  
[2] Wei Zhang, Gaoliang Peng, Chuanhao Li, Yuanhang Chen and Zhujin Zhang, “[A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaption Ability on Raw Vibration Signals](https://www.mdpi.com/1424-8220/17/2/425)”, *MDPI Sensors*, 2017.