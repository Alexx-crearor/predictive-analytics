# DeepKoopFormer — 225 runs: kết quả định lượng

Nguồn: `research_xai/results/reduced_grid_225_v1/metrics_merged.csv`. 
Giao thức: 5 datasets × M0–M4 × P={70,100,130} × H={2,8,16}, seed=7; 4.000 max epochs, batch 32.

## MSE trung bình theo dataset (9 cấu hình/model)

| Dataset | M0 | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|---:|
| Electricity | 0.02710858 | 0.02793672 | 0.0282616 | 0.02511208 | 0.02430346 |
| Financial | 0.002840549 | 0.003133967 | 0.00294564 | 0.00378263 | 0.007656735 |
| CMIP6 Wind | 0.03211399 | 0.03351243 | 0.03049749 | 0.02661777 | 0.02554659 |
| CMIP6 Pressure | 0.01007829 | 0.01017397 | 0.01046494 | 0.008984434 | 0.009399893 |
| Lorenz | 8.357963e-05 | 0.0001054647 | 9.406824e-05 | 0.001097701 | 0.0009911757 |

## M4 so với M1 (9 cặp/dataset)

Cột cải thiện được tính bằng `100 × (1 − mean(M4 MSE) / mean(M1 MSE))` trên cùng 9 tổ hợp P/H.

| Dataset | M4 thắng M1 | Cải thiện MSE trung bình |
|---|---:|---:|
| Electricity | 9/9 | +13.01% |
| Financial | 0/9 | -144.31% |
| CMIP6 Wind | 9/9 | +23.77% |
| CMIP6 Pressure | 5/9 | +7.61% |
| Lorenz | 0/9 | -839.82% |

## M4 so với M3

| Dataset | M4 thắng M3 | Cải thiện MSE trung bình |
|---|---:|---:|
| Electricity | 5/9 | +3.22% |
| Financial | 1/9 | -102.42% |
| CMIP6 Wind | 5/9 | +4.02% |
| CMIP6 Pressure | 2/9 | -4.62% |
| Lorenz | 5/9 | +9.70% |

## Early stopping

- Trung bình 1165.8/4000 epochs/run, tiết kiệm 70.86% ngân sách epochs.
- Early stopped: 225/225 runs.

## Chọn checkpoint XAI không cherry-pick

Giữ **P100/H8** cho tất cả 5 datasets; so sánh M0/M1/M3/M4 trên **cùng test window**.

| Dataset | M1 Test MSE | M4 Test MSE | M4 vs M1 |
|---|---:|---:|---:|
| Electricity | 0.02876955 | 0.02570613 | +10.6% |
| Financial | 0.002179747 | 0.008456912 | -288.0% |
| CMIP6 Wind | 0.03723591 | 0.02996535 | +19.5% |
| CMIP6 Pressure | 0.009973984 | 0.008170987 | +18.1% |
| Lorenz | 0.0001058351 | 0.0006243442 | -489.9% |

## Giới hạn và các kiểm chứng chưa hoàn thành

- Kết quả hiện có từ **1 seed**; không coi 9 cấu hình là 9 lần lặp độc lập để khẳng định ý nghĩa thống kê.
- Chỉ đánh giá grid đại diện 225 runs, không phải full grid 1.400 runs.
- CMIP6 Pressure dùng `cmip6_pressure_candidate_2500.pt`, nguồn/split cần xác minh trước khi kết luận chính thức.
- So sánh giữa kỳ cũ cần ghi rõ khác biệt preprocessing, batch, validation và scaling; không gộp MSE trực tiếp.
- XAI/latent diagnostics hiện **chưa thực hiện**; không sử dụng các cơ chế giả định để kết luận nguyên nhân thất bại.
- Không lấy trung bình MSE thô giữa các datasets có thang độ khác nhau; chuẩn hóa theo M1 hoặc phân tích riêng từng dataset.
