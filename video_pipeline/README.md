# Video → COMETH

Các script thêm vào repo, không sửa code gốc. Có hai nhánh tách riêng theo nguồn keypoint:

```
video_pipeline/
├── run_cometh.py      dùng chung: chạy IK / dynamics của COMETH trên một CSV keypoint
├── wham/              nhánh WHAM (khuyên dùng): output WHAM → IK → kiểm tra/sửa → mesh, video, bảng
│   └── out/
└── mediapipe/         nhánh MediaPipe: video → keypoint 3D bằng MediaPipe → IK
    └── out/
```

Cả hai nhánh đều xuất CSV keypoint cùng format với `inverse_dynamics/data.csv`. Sau đó cả hai dùng chung `run_cometh.py`. Hướng dẫn chi tiết từng nhánh: [wham/README.md](wham/README.md), [mediapipe/README.md](mediapipe/README.md).

## run_cometh.py (môi trường có nimblephysics, ví dụ `.venv` trong WSL)

```bash
# Inverse kinematics: qpIK + Kalman (tracker COMETH)
.venv/bin/python video_pipeline/run_cometh.py keypoints.csv --out out_ik
# IK bằng gradient descent như test_dynamics.ipynb: chậm hơn, khớp sát hơn, nên dùng cho chuyển động nhanh
.venv/bin/python video_pipeline/run_cometh.py keypoints.csv --out out_ik --ik exact
# Chạy nguyên inverse_dynamics/test_dynamics_claude.py trên CSV này
.venv/bin/python video_pipeline/run_cometh.py keypoints.csv --out out_dyn --mode dynamics
# Xem lại kết quả trên mô hình 3D: http://localhost:8080
.venv/bin/python video_pipeline/run_cometh.py keypoints.csv --out out_ik --gui
```

| File | Nội dung |
|---|---|
| `q.csv` | 49 toạ độ khớp của mô hình BSM theo thời gian (rad / m) |
| `keypoints.csv` | 12 keypoint của mô hình sau khi khớp (Z-up) và MPJPE từng frame so với input |
| `tau.csv` | (dynamics) mô-men của 43 khớp được dẫn động [Nm] |
| `grf.csv` | (dynamics) số điểm tiếp xúc, lực phản lực đất theo phương đứng [N] và tỉ lệ so với trọng lượng cơ thể |
| `joint_angles.png`, `dynamics.png` | biểu đồ nhanh |

**`--ik exact` khác `--ik qp` ở đâu:**
- `qp` có giới hạn vận tốc học từ dữ liệu đi bộ, cộng thêm bộ lọc Kalman. Nhờ vậy rất mượt, nhưng không theo kịp chuyển động nhanh: với cú nhảy trượt băng, nó lệch keypoint khoảng 29 cm.
- `exact` khớp từng frame, không có ràng buộc theo thời gian. Bám sát dữ liệu hơn (khoảng 22 mm) nhưng góc có thể giật giữa các frame.

**Chế độ `dynamics`:**
- Resample input về dt = 0.033 s, vì giá trị này đang hardcode trong `test_dynamics_claude.py`.
- Sau đó chạy nguyên script đó, chỉ chuyển hướng đường dẫn `data.csv` sang file của bạn.
- **Giới hạn của code gốc:**
  - Crash khi trong một frame không có điểm bàn chân nào cao dưới 5 cm (`np.vstack` trên danh sách rỗng). Vì vậy nó không xử lý được pha bay hay trường hợp mặt đất bị ước lượng lệch.
  - Có thể dừng giữa chừng khi OSQP fail.
  - Mô-men bị kẹp ở ±100 Nm.
- Với dữ liệu WHAM, nên dùng `wham/dynamics_report.py`.
