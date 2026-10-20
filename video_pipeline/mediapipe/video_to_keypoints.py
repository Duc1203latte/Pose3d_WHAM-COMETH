"""
Video -> CSV keypoint 3D cùng format với inverse_dynamics/data.csv.

Chạy MediaPipe PoseLandmarker trên video một camera, giữ lại 12 khớp BODY12 mà COMETH
dùng và ghi ra theo đơn vị mét, trục Y hướng lên (các script trong inverse_dynamics/
tự xoay Y-up -> Z-up).

Bước này KHÔNG cần nimblephysics, nên chạy trực tiếp được trên Windows:
    pip install mediapipe opencv-python pandas
    python video_pipeline/mediapipe/video_to_keypoints.py my_video.mp4 -o video_pipeline/mediapipe/out/my_video.csv \
        --vis video_pipeline/mediapipe/out/my_video_vis.mp4

Giới hạn của video một camera:
    - World landmark của MediaPipe lấy gốc ở hông: mất quỹ đạo toàn cục của người
      (cơ thể như "đi tại chỗ").
    - Không quan sát được mặt đất: mặt đất được đặt bằng cách đưa điểm bàn chân thấp nhất
      về y=0 (xem --ground).
"""
import argparse
import os
import urllib.request

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# tiền tố cột của data.csv -> chỉ số landmark MediaPipe
JOINTS = {
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
    "right_hip": 24,
    "right_knee": 26,
    "right_ankle": 28,
    "left_hip": 23,
    "left_knee": 25,
    "left_ankle": 27,
}
# Cổ chân, gót và mũi chân: dùng để tìm mặt đất
FOOT_LANDMARKS = [27, 28, 29, 30, 31, 32]

BONES = [
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "left_shoulder"), ("right_hip", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_shoulder", "left_hip"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
]


def get_model(model_path, variant):
    if model_path is not None:
        return model_path
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "pose_landmarker_{}.task".format(variant))
    if not os.path.exists(path):
        print("Downloading", MODEL_URLS[variant])
        urllib.request.urlretrieve(MODEL_URLS[variant], path)
    return path


def run_mediapipe(video_path, model_path, min_conf, vis_path=None):
    """
    Trả về:
        world: (N, 33, 3) world landmark theo quy ước MediaPipe (x sang phải, y xuống dưới, z theo chiều sâu),
               NaN nếu không phát hiện được người
        fps: frame rate của video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_conf,
        min_pose_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )

    writer = None
    world = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(round(1000 * i / fps)))

            if result.pose_world_landmarks:
                world.append([[l.x, l.y, l.z] for l in result.pose_world_landmarks[0]])
            else:
                world.append(np.full((33, 3), np.nan).tolist())

            if vis_path is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    os.makedirs(os.path.dirname(os.path.abspath(vis_path)), exist_ok=True)
                    writer = cv2.VideoWriter(vis_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if result.pose_landmarks:
                    draw_2d(frame, result.pose_landmarks[0])
                writer.write(frame)

            i += 1
            if i % 50 == 0:
                print("  frame {}/{}".format(i, n_frames))
    cap.release()
    if writer is not None:
        writer.release()
    return np.array(world, dtype=float), fps


def draw_2d(frame, landmarks):
    h, w = frame.shape[:2]
    px = {name: (int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for name, idx in JOINTS.items()}
    for a, b in BONES:
        cv2.line(frame, px[a], px[b], (0, 255, 0), 2)
    for p in px.values():
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def fill_gaps(x):
    """Nội suy tuyến tính qua các frame không phát hiện được pose; (N, J, 3)."""
    n = x.shape[0]
    t = np.arange(n)
    flat = x.reshape(n, -1)
    for c in range(flat.shape[1]):
        valid = ~np.isnan(flat[:, c])
        if valid.any() and not valid.all():
            flat[:, c] = np.interp(t, t[valid], flat[valid, c])
    return flat.reshape(x.shape)


def smooth(x, window):
    """Trung bình trượt đối xứng theo thời gian (đệm bằng giá trị ở hai đầu); (N, J, 3)."""
    if window <= 1:
        return x
    pad = window // 2
    padded = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], 0)
    kernel = np.ones(window) / window
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        for k in range(3):
            out[:, j, k] = np.convolve(padded[:, j, k], kernel, mode="valid")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("-o", "--output", default=None, help="CSV output (mặc định: <video>.csv)")
    parser.add_argument("--vis", default=None, help="(tuỳ chọn) video output có vẽ skeleton 2D phát hiện được")
    parser.add_argument("--model", default=None, help="đường dẫn model .task (mặc định: tải theo --variant)")
    parser.add_argument("--variant", default="heavy", choices=list(MODEL_URLS.keys()))
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--smooth", type=int, default=5, help="cửa sổ trung bình trượt, tính bằng frame (1 = tắt)")
    parser.add_argument("--ground", default="frame", choices=["frame", "sequence", "none"],
                        help="frame: điểm bàn chân thấp nhất ở y=0 trong mỗi frame; sequence: điểm bàn chân thấp "
                             "nhất của cả video ở y=0; none: giữ gốc toạ độ của MediaPipe (hông)")
    args = parser.parse_args()

    output = args.output or os.path.splitext(args.video)[0] + ".csv"
    world, fps = run_mediapipe(args.video, get_model(args.model, args.variant), args.min_conf, args.vis)

    detected = ~np.isnan(world[:, 0, 0])
    print("Detected a person in {}/{} frames ({:.1f} fps)".format(detected.sum(), len(world), fps))
    if not detected.any():
        raise RuntimeError("No person detected in the video")
    # Bỏ các frame không phát hiện được ở đầu và cuối, nội suy các lỗ ở giữa
    first, last = np.where(detected)[0][[0, -1]]
    world = fill_gaps(world[first:last + 1])
    world = smooth(world, args.smooth)

    # MediaPipe (x sang phải, y xuống dưới, z ra xa camera) -> hệ tay phải Y-up (xoay 180 độ quanh trục x)
    world = world * np.array([1.0, -1.0, -1.0])

    if args.ground == "frame":
        world[:, :, 1] -= world[:, FOOT_LANDMARKS, 1].min(axis=1, keepdims=True)
    elif args.ground == "sequence":
        world[:, :, 1] -= world[:, FOOT_LANDMARKS, 1].min()

    frame_ids = np.arange(first, last + 1)
    df = pd.DataFrame({
        "timestamp": np.round(frame_ids / fps * 1e9).astype(np.int64),  # ns, như trong data.csv
        "frame_id": frame_ids,
        "body_id": 0,
    })
    for name, idx in JOINTS.items():
        for k, axis in enumerate("xyz"):
            df["{}_{}".format(name, axis)] = world[:, idx, k]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df.to_csv(output)
    print("Saved {} frames to {}".format(len(df), output))


if __name__ == "__main__":
    main()
