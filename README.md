# PytorchCrf

基於PyTorch實做的CRF模型。

因PyTorch官方並無直接支援CRF但有提供實作CRF的程式碼，但是在速度上不太理想。
研究後發現官方提供的CRF速度慢是因為沒有使用矩陣運算，只單純使用元素運算。
改寫成矩陣運算後，讓執行時間從s單位進步到ms單位。

## 介紹

`Bilstm.py`：簡單的BiLSTM模型，包含Embedding與Linear。
`Crf.py`：自製的CRF層
`BilstmCrf.py`：組合BiLSTM+CRF模型
`Trainer.py`：訓練模組
