"""
Output WHAM (wham_output.pkl) -> CSV keypoint 3D cùng format với inverse_dynamics/data.csv.

WHAM cho mesh SMPL trong hệ camera ('verts'), cùng hướng / vị trí gốc trong hệ world thẳng hàng với
trọng lực ('pose_world', 'trans_world'). 24 khớp SMPL được tính từ mesh bằng J_regressor của SMPL rồi
chuyển sang hệ world - quỹ đạo toàn cục của người được giữ lại.

Output: 12 khớp BODY12, đơn vị mét, trục Y hướng lên (hệ world của WHAM), mặt đất ở y=0.

Cần file model SMPL mà WHAM dùng (SMPL_NEUTRAL.pkl, mặc định ở DEFAULT_SMPL) và chumpy để đọc nó (có sẵn trong env WHAM):
    python video_pipeline/wham/wham_to_keypoints.py path/to/wham_output.pkl \
        --fps 60 -o video_pipeline/wham/out/x/wham_keypoints.csv [--smpl-model path/to/SMPL_NEUTRAL.pkl]
"""
import argparse
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial.transform import Rotation

# tiền tố cột của data.csv -> chỉ số khớp SMPL
JOINTS = {
    "right_shoulder": 17,
    "right_elbow": 19,
    "right_wrist": 21,
    "left_shoulder": 16,
    "left_elbow": 18,
    "left_wrist": 20,
    "right_hip": 2,
    "right_knee": 5,
    "right_ankle": 8,
    "left_hip": 1,
    "left_knee": 4,
    "left_ankle": 7,
}
PELVIS = 0
# Chỗ WHAM đặt model SMPL (fetch_demo_data.sh của WHAM tải về đây)
DEFAULT_SMPL = os.path.expanduser("~/projects/WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl")


def _dense(x):
    x = x.toarray() if scipy.sparse.issparse(x) else x
    return np.asarray(getattr(x, "r", x))  # mảng chumpy -> numpy


def load_smpl(path):
    """
    File model SMPL (.pkl gốc của SMPL / SMPLify) -> J_regressor (24, 6890), skin weights (6890, 24),
    faces (13776, 3). Đây là đúng model mà WHAM dùng để sinh 'verts'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Không thấy model SMPL: {} (tải bằng fetch_demo_data.sh của WHAM, xem "
                                "video_pipeline/wham/README.md)".format(path))
    with open(path, "rb") as f:
        m = pickle.load(f, encoding="latin1")
    J, W, F = _dense(m["J_regressor"]), _dense(m["weights"]), _dense(m["f"]).astype(np.int64)
    if J.shape != (24, 6890) or W.shape != (6890, 24):
        raise ValueError("Không phải model SMPL: J_regressor {}, weights {}".format(J.shape, W.shape))
    return J, W, F


def to_world(points_cam, pelvis_cam, R_wc, pelvis_world):
    """Dời cứng các điểm (N, P, 3) trong hệ camera sao cho khung chậu trùng với pelvis_world."""
    rel = points_cam - pelvis_cam[:, None]
    return np.einsum("nij,npj->npi", R_wc, rel) + pelvis_world[:, None]


def to_camera(points_world, pelvis_cam, R_wc, pelvis_world):
    """Phép ngược của to_world."""
    rel = points_world - pelvis_world[:, None]
    return np.einsum("nji,npj->npi", R_wc, rel) + pelvis_cam[:, None]


def camera_to_world(d, J, ground="sequence"):
    """
    Phép biến đổi cứng theo từng frame giữa hệ camera của WHAM và hệ của CSV (world WHAM, mặt đất ở y=0).
    Trả về joints_cam (N, 24, 3), R_wc (N, 3, 3), pelvis_cam (N, 3), pelvis_world (N, 3) và độ cao mặt đất
    (N,) cần trừ khỏi toạ độ y trong hệ world.
    """
    verts_cam = d["verts"].astype(np.float64)
    joints_cam = np.einsum("jv,nvc->njc", J, verts_cam)

    # Phép xoay camera -> world: R_world_root * R_cam_root^-1
    R_wc = (Rotation.from_rotvec(d["pose_world"][:, :3]) * Rotation.from_rotvec(d["pose"][:, :3]).inv()).as_matrix()
    # SMPL: khung chậu = J0(betas) + trans, cùng một J0 ở cả hai hệ
    pelvis_cam = joints_cam[:, PELVIS]
    pelvis_world = d["trans_world"] + (pelvis_cam - d["trans"])

    lowest = to_world(verts_cam, pelvis_cam, R_wc, pelvis_world)[:, :, 1].min(axis=1)
    if ground == "sequence":
        floor = np.full(len(lowest), np.percentile(lowest, 10))
    elif ground == "frame":
        floor = lowest
    else:
        floor = np.zeros(len(lowest))
    return joints_cam, R_wc, pelvis_cam, pelvis_world, floor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wham_pkl")
    parser.add_argument("--smpl-model", default=DEFAULT_SMPL, help="SMPL_NEUTRAL.pkl mà WHAM dùng")
    parser.add_argument("--fps", type=float, required=True, help="frame rate của video đã đưa vào WHAM")
    parser.add_argument("--subject", type=int, default=None, help="id track của WHAM (mặc định: track dài nhất)")
    parser.add_argument("-o", "--output", default=None, help="CSV output (mặc định: cạnh file pkl)")
    parser.add_argument("--ground", default="sequence", choices=["sequence", "frame", "none"],
                        help="sequence: mặt đất = phân vị 10 của đỉnh mesh thấp nhất mỗi frame; frame: đỉnh thấp "
                             "nhất ở y=0 trong mỗi frame; none: giữ nguyên độ cao của WHAM")
    args = parser.parse_args()

    results = joblib.load(args.wham_pkl)
    subject = args.subject if args.subject is not None else max(results, key=lambda k: len(results[k]["frame_ids"]))
    d = results[subject]
    print("WHAM tracks: {} -> using id {} ({} frames)".format(
        [int(k) for k in results.keys()], int(subject), len(d["frame_ids"])))

    J, _, _ = load_smpl(args.smpl_model)
    joints_cam, R_wc, pelvis_cam, pelvis_world, floor = camera_to_world(d, J, args.ground)
    joints = to_world(joints_cam, pelvis_cam, R_wc, pelvis_world)
    joints[:, :, 1] -= floor[:, None]

    frame_ids = np.asarray(d["frame_ids"])
    df = pd.DataFrame({
        "timestamp": np.round(frame_ids / args.fps * 1e9).astype(np.int64),  # ns, như trong data.csv
        "frame_id": frame_ids,
        "body_id": int(subject),
    })
    for name, idx in JOINTS.items():
        for k, axis in enumerate("xyz"):
            df["{}_{}".format(name, axis)] = joints[:, idx, k]

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.wham_pkl)), "wham_keypoints.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df.to_csv(output)
    travel = np.linalg.norm(np.diff(pelvis_world[:, [0, 2]], axis=0), axis=1).sum()
    print("Saved {} frames to {}  (pelvis horizontal travel {:.2f} m)".format(len(df), output, travel))


if __name__ == "__main__":
    main()
