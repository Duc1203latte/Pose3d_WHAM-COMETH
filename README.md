# Pose3d WHAM-COMETH

Phân tích tư thế 3D từ **video một camera**:
- lấy chuyển động người từ [WHAM](https://github.com/yohanshin/WHAM) (hoặc MediaPipe);
- khớp vào mô hình xương sinh học BSM bằng [COMETH](https://github.com/PARCO-LAB/COMETH);
- kiểm tra và sửa các góc khớp không hợp lý, xuất lại mesh / video;
- lập bảng đánh dấu các giá trị bất thường, gồm cả mô-men và lực khi chạy inverse dynamics.

## Chức năng

| | Script (trong `video_pipeline/`) |
|---|---|
| Output WHAM → keypoint 3D | `wham/wham_to_keypoints.py` |
| Keypoint → góc khớp BSM (IK của COMETH) | `run_cometh.py` |
| Kiểm tra / sửa góc khớp (ROM, đột biến, vận tốc), bảng `report.html` | `wham/refine_q.py` |
| Skeleton trước / sau khi sửa vẽ lên video gốc | `wham/overlay_video.py` |
| Áp phần sửa vào mesh SMPL của WHAM | `wham/wham_refine_mesh.py` |
| Video kiểu WHAM: bản sửa, so sánh, chồng hai mesh | `wham/render_wham.py` |
| Bảng "đã sửa gì" (`changes.html`) | `wham/change_report.py` |
| Inverse dynamics (mô-men, lực đất, lực dư) + bảng đánh dấu | `wham/dynamics_report.py` |
| Video → keypoint 3D bằng MediaPipe | `mediapipe/video_to_keypoints.py` |

Hướng dẫn chi tiết:
- [video_pipeline/README.md](video_pipeline/README.md): tổng quan và `run_cometh.py`
- [video_pipeline/wham/README.md](video_pipeline/wham/README.md): nhánh WHAM
- [video_pipeline/mediapipe/README.md](video_pipeline/mediapipe/README.md): nhánh MediaPipe

## Môi trường

- **`.venv`** (Linux / WSL, có nimblephysics, cvxpy): IK, sửa góc khớp, inverse dynamics. Cài theo `requirements.txt` và `pip install -e .`.
- **Env của WHAM** (torch, pytorch3d, opencv, chumpy): đọc output WHAM, xử lý mesh, render.
- **Model SMPL** `SMPL_NEUTRAL.pkl`, đúng model WHAM dùng. Tải tại https://smplify.is.tue.mpg.de (cần đăng ký), xem [video_pipeline/wham/README.md](video_pipeline/wham/README.md).

## Nguồn gốc và giấy phép

Repo này được xây dựng trên **COMETH** (Martini et al., *Expert Systems with Applications*, 2026, [paper](https://doi.org/10.1016/j.eswa.2026.131728), [repo gốc](https://github.com/PARCO-LAB/COMETH)):
- Các thư mục `COMETH/`, `inverse_dynamics/`, `multiview_skeleton_fusion/`, `IMU-HPE_fusion/`, `sf_utils/`, `utils/` là code gốc của COMETH. README gốc nằm ở [COMETH_README.md](COMETH_README.md).
- Phần bổ sung của repo này nằm trong `video_pipeline/`.

Phát hành theo giấy phép GPL-3.0, giống repo gốc. Xem [LICENSE](LICENSE).
