# Nhánh MediaPipe: video → keypoint 3D → COMETH

```
video.mp4 ──[video_to_keypoints.py]──► keypoints.csv ──[../run_cometh.py]──► q.csv, keypoints.csv, ...
           (Windows/Linux, MediaPipe)   (cùng format data.csv)  (WSL/Linux, nimblephysics)
```

## Video → CSV keypoint 3D

```bash
pip install mediapipe opencv-python pandas
python video_pipeline/mediapipe/video_to_keypoints.py my_video.mp4 -o video_pipeline/mediapipe/out/my_video.csv \
    --vis video_pipeline/mediapipe/out/my_video_vis.mp4
```

- Model MediaPipe PoseLandmarker (`--variant heavy|full|lite`) được tải tự động vào `video_pipeline/mediapipe/models/`.
- Giữ lại 12 khớp BODY12, đơn vị là mét, trục Y hướng lên, tên cột giống `inverse_dynamics/data.csv`.
- `--ground`:
  - `frame` (mặc định): ở mỗi frame, điểm thấp nhất của bàn chân được đặt ở y = 0.
  - `sequence`: chỉ dời một lần cho cả video.
  - `none`: giữ gốc toạ độ ở hông như MediaPipe.
- `--smooth N`: làm mượt bằng trung bình trượt N frame. Frame không phát hiện được người sẽ được nội suy.
- `--vis`: xuất video có vẽ skeleton 2D để kiểm tra detection.

## Chạy COMETH

```bash
.venv/bin/python video_pipeline/run_cometh.py video_pipeline/mediapipe/out/my_video.csv \
    --out video_pipeline/mediapipe/out/my_video_ik --ik exact
```

Xem các chế độ và output trong [../README.md](../README.md).

## Giới hạn

- **Không có quỹ đạo toàn cục:** MediaPipe đặt gốc ở hông, nên người trong kết quả như đang di chuyển tại chỗ.
- **Mặt đất là ước lượng:** vì thế lực tiếp xúc và mô-men chỉ mang tính tham khảo.
- **Chuyển động xoay nhanh hoặc quay lưng camera** (ví dụ cú nhảy trượt băng): MediaPipe nhầm trái/phải và hướng thân nhảy cóc giữa các frame. Khi đó IK có sai số lớn, và QP dynamics có thể fail ngay từ frame đầu.
- **Nên dùng với:** video đi lại, squat, nâng tay..., người quay mặt hoặc nghiêng về phía camera. Với chuyển động thể thao, dùng nhánh [WHAM](../wham/README.md).

`out/toeloop_mediapipe/` là lần thử với clip toe loop, minh hoạ đúng các giới hạn trên.
