"""
Đưa phần sửa của refine_q.py vào mesh SMPL của WHAM, rồi lưu thành một wham_output.pkl mới.

Body pose SMPL (23 khớp) được fit lại sao cho 12 keypoint SMPL bám theo skeleton COMETH đã sửa. Sau đó chính
mesh WHAM được biến dạng bằng linear blend skinning *tương đối* so với pose gốc của nó
(v' = sum_k w_k * M_k * v, M_k = new_k * orig_k^-1), với skin weights và J_regressor của chính model SMPL mà
WHAM dùng (SMPL_NEUTRAL.pkl). Nhờ vậy người giữ nguyên đúng shape của WHAM.
Không áp lại pose blend shapes (khác biệt nhỏ quanh các khớp bị gập).

Mục tiêu của 12 keypoint, trong hệ của CSV (xem wham_to_keypoints.py):
    keypoint COMETH đã sửa + (keypoint SMPL - keypoint COMETH gốc) đã lọc thông thấp
tức là skeleton đã sửa, cộng với độ lệch cố định (đã làm mượt) giữa tâm khớp của SMPL và của BSM.

Cần torch (+ numpy, scipy, joblib): ví dụ env WHAM.
    python video_pipeline/wham/wham_refine_mesh.py path/to/wham_output.pkl \
        --refined video_pipeline/wham/out/x/ik_exact/refined [--smpl-model path/to/SMPL_NEUTRAL.pkl]

Output: <refined>/wham_output_refined.pkl, cùng cấu trúc với output WHAM (cập nhật pose, pose_world, verts;
giữ nguyên trans, trans_world, betas, frame_ids). Render bằng render_wham.py.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt

from wham_to_keypoints import DEFAULT_SMPL, JOINTS, camera_to_world, load_smpl, to_camera, to_world

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
# keypoint COMETH -> tiền tố cột của CSV keypoint
KP_CSV = {
    "RShoulder": "right_shoulder", "RElbow": "right_elbow", "RWrist": "right_wrist",
    "LShoulder": "left_shoulder", "LElbow": "left_elbow", "LWrist": "left_wrist",
    "RHip": "right_hip", "RKnee": "right_knee", "RAnkle": "right_ankle",
    "LHip": "left_hip", "LKnee": "left_knee", "LAnkle": "left_ankle",
}
KPS = list(KP_CSV)
SMPL_IDX = [JOINTS[KP_CSV[k]] for k in KPS]


def axis_angle_to_matrix(aa):
    """(..., 3) -> (..., 3, 3), công thức Rodrigues."""
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    Kx = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(aa.shape[:-1] + (3, 3))
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(Kx)
    return eye + s * Kx + (1 - c) * Kx @ Kx


def global_rotations(R_local):
    """(N, 24, 3, 3) phép xoay cục bộ -> phép xoay toàn cục dọc theo cây khớp SMPL."""
    G = [R_local[:, 0]]
    for k in range(1, 24):
        G.append(G[SMPL_PARENTS[k]] @ R_local[:, k])
    return torch.stack(G, dim=1)


def relative_transforms(pose0, pose, joints0):
    """
    Phép biến đổi cứng của từng phần cơ thể SMPL từ pose gốc sang pose mới, xoay quanh các khớp ở tư thế hiện tại.
    pose0, pose: (N, 24, 3) axis-angle; joints0: (N, 24, 3) khớp của mesh gốc ở tư thế hiện tại.
    Trả về dR (N, 24, 3, 3) và vị trí khớp mới (N, 24, 3): x' = dR_k (x - joints0_k) + joints_k.
    """
    G0 = global_rotations(axis_angle_to_matrix(pose0))
    G = global_rotations(axis_angle_to_matrix(pose))
    dR = G @ G0.transpose(-1, -2)
    joints = [joints0[:, 0]]
    for k in range(1, 24):
        p = SMPL_PARENTS[k]
        joints.append(joints[p] + (dR[:, p] @ (joints0[:, k] - joints0[:, p])[..., None])[..., 0])
    return dR, torch.stack(joints, dim=1)


def skin(verts0, W, dR, joints0, joints):
    """LBS tương đối trên các đỉnh của mesh gốc (N, V, 3)."""
    local = verts0[:, :, None] - joints0[:, None]                                   # (N, V, 24, 3)
    moved = torch.einsum("nkij,nvkj->nvki", dR, local) + joints[:, None]             # (N, V, 24, 3)
    return torch.einsum("vk,nvki->nvi", W, moved)


def lowpass(x, fs, cutoff):
    if cutoff <= 0 or len(x) <= 15:
        return x
    b, a = butter(2, min(cutoff / (fs / 2), 0.99))
    return filtfilt(b, a, x, axis=0)


def load_model_joints(path, frame_ids):
    df = pd.read_csv(path).set_index("frame_id").loc[frame_ids]
    return np.stack([df[["{}_{}".format(k, a) for a in "xyz"]].values for k in KPS], axis=1), df["time"].values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wham_pkl")
    parser.add_argument("--smpl-model", default=DEFAULT_SMPL, help="SMPL_NEUTRAL.pkl, giống lúc chạy wham_to_keypoints.py")
    parser.add_argument("--refined", required=True, help="thư mục output của refine_q.py")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--ground", default="sequence", choices=["sequence", "frame", "none"])
    parser.add_argument("--offset-cutoff", type=float, default=3.0,
                        help="[Hz] lọc thông thấp độ lệch keypoint SMPL - BSM (0: giữ nguyên)")
    parser.add_argument("--w-prior", type=float, default=1e-3,
                        help="trọng số của |pose - pose WHAM|^2 (giữ các DOF mà 12 keypoint không quan sát được)")
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-o", "--output", default=None, help="mặc định: <refined>/wham_output_refined.pkl")
    args = parser.parse_args()

    results = joblib.load(args.wham_pkl)
    subject = args.subject if args.subject is not None else max(results, key=lambda k: len(results[k]["frame_ids"]))
    d = results[subject]
    frame_ids = np.asarray(d["frame_ids"])

    J, W, _ = load_smpl(args.smpl_model)
    joints_cam, R_wc, pelvis_cam, pelvis_world, floor = camera_to_world(d, J, args.ground)

    # Mục tiêu trong hệ của CSV, rồi chuyển sang hệ camera
    smpl_csv = to_world(joints_cam[:, SMPL_IDX], pelvis_cam, R_wc, pelvis_world)
    smpl_csv[:, :, 1] -= floor[:, None]
    before, times = load_model_joints(os.path.join(args.refined, "joints_before.csv"), frame_ids)
    after, _ = load_model_joints(os.path.join(args.refined, "joints_refined.csv"), frame_ids)
    fs = 1.0 / np.median(np.diff(times))
    offset = lowpass((smpl_csv - before).reshape(len(frame_ids), -1), fs, args.offset_cutoff).reshape(before.shape)
    target = after + offset
    target[:, :, 1] += floor[:, None]
    target_cam = to_camera(target, pelvis_cam, R_wc, pelvis_world)

    dev = torch.device(args.device)
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
    pose0 = tt(d["pose"]).reshape(-1, 24, 3)
    joints0 = tt(joints_cam)
    target_t = tt(target_cam)
    body = pose0[:, 1:].clone().requires_grad_(True)
    opt = torch.optim.Adam([body], lr=0.01)
    for it in range(args.iters):
        pose = torch.cat([pose0[:, :1], body], dim=1)
        _, joints = relative_transforms(pose0, pose, joints0)
        fit = ((joints[:, SMPL_IDX] - target_t) ** 2).sum(-1).mean()
        prior = ((body - pose0[:, 1:]) ** 2).sum(-1).mean()
        loss = fit + args.w_prior * prior
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it == 0 or (it + 1) % 100 == 0:
            print("  iter {:4d}  keypoint error {:.1f} mm".format(it + 1, 1000 * fit.sqrt().item()))

    with torch.no_grad():
        pose = torch.cat([pose0[:, :1], body], dim=1)
        dR, joints = relative_transforms(pose0, pose, joints0)
        verts = torch.cat([skin(tt(d["verts"][i:i + 32]), tt(W), dR[i:i + 32], joints0[i:i + 32], joints[i:i + 32])
                           for i in range(0, len(frame_ids), 32)])
        err0 = (joints0[:, SMPL_IDX] - target_t).norm(dim=-1).mean(-1).cpu().numpy()
        err = (joints[:, SMPL_IDX] - target_t).norm(dim=-1).mean(-1).cpu().numpy()
        moved = (verts - tt(d["verts"])).norm(dim=-1).mean(-1).cpu().numpy()

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    body_np = pose[:, 1:].reshape(len(frame_ids), -1).cpu().numpy()
    out["pose"][:, 3:] = body_np
    out["pose_world"][:, 3:] = body_np
    out["verts"] = verts.cpu().numpy().astype(d["verts"].dtype)
    output = args.output or os.path.join(args.refined, "wham_output_refined.pkl")
    joblib.dump({subject: out}, output)

    print("Distance of the SMPL keypoints to the targets: {:.1f} mm (WHAM) -> {:.1f} mm (refined mesh)".format(
        1000 * err0.mean(), 1000 * err.mean()))
    print("Mean vertex displacement: {:.1f} mm (max frame {:.1f} mm, frame id {})".format(
        1000 * moved.mean(), 1000 * moved.max(), frame_ids[np.argmax(moved)]))
    print("Saved", output)


if __name__ == "__main__":
    main()
