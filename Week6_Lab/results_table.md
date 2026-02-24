# Hyperparameter Tuning Results

| Config | Changes Made                                            | Time (min)  | Train Acc (%) | Val Acc (%) | Test Acc (%) | Test Loss | Notes |
|--------|---------------------------------------------------------|-------------|--------------:|------------:|-------------:|----------:|-------|
| 1      | filters=(32,64,128), dropout=0.5, epochs=10, bs=32, lr=0.001|6.22           |77.10%       |78.00%        | 81.27%  |0.4985 |Underfitting
| 2      | filters=(64,128,256), dropout=0.5,epochs=10, bs=32, lr=0.001|6.24           |78.17%       |77.89%        | 79.47%  |0.6042 |Underfitting          |
| 3      | filters=(64,128,256), dropout=0.7, epochs=20 bs=32, lr=0.001|12.43   _      |79.85%       |78.96%        | 81.80%  |0.5576 |           |
| 4      | filters=(32,64,128), dropout=0.7, epochs=40 bs=32,  lr=0.001|23.41          |82.17%       |82.49%        | 84.27%  |0.4596 |Mild Overfit           |
| 5      | filters=(32,64,128), dropout=0.7, epochs=40 bs=64,  lr=0.001|24.01          |83.59%       |85.31%        | 85.50%  |0.4176 |Balanced           |
| 6      | filters=(64,128,256), dropout=0.7, epochs=40 bs=64, lr=0.001|25.10          |83.48%       |84.88%        | 86.33%  |0.3853 |
| 7      | filters=(32,64,128), dropout=0.7, epochs=40, bs=128,lr=0.001|24.36          |81.05%       |83.59%        | 85.77%  |0.3997 |
| 8      | filters=(64,128,256), dropout=0.5, epochs=60, bs=32,lr=0.001|31.50          |85.62%       |85.41%        |86.33%   |0.4095 | Best
