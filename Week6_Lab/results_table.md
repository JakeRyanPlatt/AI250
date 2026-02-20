# Hyperparameter Tuning Results

| Config | Changes Made                                            | Time (min)  | Train Acc (%) | Val Acc (%) | Test Acc (%) | Test Loss | Notes |
|--------|---------------------------------------------------------|-------------|--------------:|------------:|-------------:|----------:|-------|
| 1      | Baseline: filters=(32,64,128), dropout=0.5, epochs=10, bs=32, lr=0.001|5.82           |77.10%       |78.00%        |   81.27%  |       |
| 2      | filters=(64,128,256)                                    |             |               |             |              |           |       |
| 3      | filters=(64,128,256), dropout=0.7, epochs=20            |             |               |             |              |           |       |
