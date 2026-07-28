# -*- coding: utf-8 -*-
"""
1DCNN 模型訓練與測試腳本 (對應 1DCNN_demo.ipynb)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def main():
    print("[1DCNN 訓練] 載入數據集 ./data/ABCD_Datasets.pickle ...")
    pfile = os.path.join(".", "data", "ABCD_Datasets.pickle")
    if not os.path.exists(pfile):
        raise FileNotFoundError(f"找不到數據檔: {pfile}，請先執行 run_data_prep.py！")

    with open(pfile, "rb") as openfile:
        ABCD_Datasets = pickle.load(openfile)

    X_train_D = ABCD_Datasets["train_datasets"]["D"]
    Y_train_D = ABCD_Datasets["train_labels"]["D"]
    X_test_D = ABCD_Datasets["test_datasets"]["D"]
    Y_test_D = ABCD_Datasets["test_labels"]["D"]

    num_classes = 10
    Y_train_D_hot = keras.utils.to_categorical(Y_train_D - 1, num_classes)
    Y_test_D_hot = keras.utils.to_categorical(Y_test_D - 1, num_classes)

    (X_train, X_valid) = X_train_D[2000:], X_train_D[:2000]
    (Y_train, Y_valid) = Y_train_D_hot[2000:], Y_train_D_hot[:2000]
    X_test = X_test_D
    Y_test = Y_test_D_hot

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_valid.shape[0], 'validation samples')

    print("\n--- 1. 邏輯回歸 Baseline 模型 (Logistic Regression) ---")
    CNNch_log = 1
    ne = 10

    modelLog = Sequential()
    modelLog.add(Flatten(input_shape=(2048, CNNch_log)))
    modelLog.add(Dense(10, activation='softmax'))
    modelLog.summary()

    modelLog.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    modelLog.fit(
        X_train[:, :, 0:CNNch_log], Y_train,
        batch_size=32, epochs=ne,
        validation_data=(X_valid[:, :, 0:CNNch_log], Y_valid),
        verbose=1, shuffle=True
    )

    scoreLog = modelLog.evaluate(X_test[:, :, 0:CNNch_log], Y_test, verbose=0)
    print('\nLogistic Regression Test accuracy:', scoreLog[1])

    print("\n--- 2. 1D-CNN 特徵模型 (2-Channel 1D Conv) ---")
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

    modelC2.summary()

    modelC2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    weights_filepath = 'CNNC2.weights.best.weights.h5'
    checkpointer = ModelCheckpoint(filepath=weights_filepath, verbose=1, save_best_only=True, save_weights_only=True)

    histC2 = modelC2.fit(
        X_train[:, :, 0:CNNch], Y_train,
        batch_size=32, epochs=ne,
        validation_data=(X_valid[:, :, 0:CNNch], Y_valid),
        callbacks=[checkpointer],
        verbose=1, shuffle=True
    )

    modelC2.load_weights(weights_filepath)

    score_test = modelC2.evaluate(X_test[:, :, 0:CNNch], Y_test, verbose=0)
    print('\nCNN Test accuracy:', score_test[1])

    score_train = modelC2.evaluate(X_train[:, :, 0:CNNch], Y_train, verbose=0)
    print('\nCNN train accuracy:', score_train[1])

    score_valid = modelC2.evaluate(X_valid[:, :, 0:CNNch], Y_valid, verbose=0)
    print('\nCNN validation accuracy:', score_valid[1])

    print("\n--- 3. 繪製並保存 1D-CNN 訓練曲線與測試集混淆矩陣 ---")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(histC2.history['loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(histC2.history['val_loss'], label='Val Loss', color='red', linestyle='--', linewidth=2)
    plt.title('1D-CNN Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(histC2.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(histC2.history['val_accuracy'], label='Val Accuracy', color='green', linestyle='--', linewidth=2)
    plt.title('1D-CNN Accuracy Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('1DCNN_training_history.png', dpi=300)
    print("訓練曲線圖已成功保存至 1DCNN_training_history.png")
    plt.close()

    Y_pred = modelC2.predict(X_test[:, :, 0:CNNch])
    y_pred_classes = np.argmax(Y_pred, axis=1)
    y_true_classes = np.argmax(Y_test, axis=1)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_classes, y_pred_classes):
        cm[t, p] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('1D-CNN Test Set Confusion Matrix', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    class_names = [f'Class {i+1}' for i in range(num_classes)]
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('1DCNN_confusion_matrix.png', dpi=300)
    print("混淆矩陣圖已成功保存至 1DCNN_confusion_matrix.png")
    plt.close()

if __name__ == '__main__':
    main()
