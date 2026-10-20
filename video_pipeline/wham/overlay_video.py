"""
Vẽ skeleton COMETH trước / sau refine_q.py lên video gốc bằng camera của WHAM, và đánh dấu các khớp có góc
bị gắn cờ không hợp lý.

Cần numpy, pandas, scipy, joblib và opencv (ví dụ env WHAM, không cần nimblephysics):
    python video_pipeline/wham/overlay_video.py path/to/wham_output.pkl --video original.mp4 \
        --refined video_pipeline/wham/out/x/ik_exact/refined [--smpl-model path/to/SMPL_NEUTRAL.pkl]

--video là video đã đưa vào WHAM (hoặc video bất kỳ có cùng frame và độ phân giải); thông số nội của camera
đọc từ calib.txt cạnh file pkl (fx fy cx cy, do WHAM ghi). --ground và --subject phải giống lúc chạy
wham_to_keypoints.py.

Bố cục:
    side     hai khung: trước (xanh dương) | sau (xanh lá), mỗi khung có cờ riêng      (mặc định)
    overlay  một khung, cả hai skeleton, cờ của góc gốc
Skeleton trắng mảnh là input WHAM. Vòng tròn đánh dấu khớp bị gắn cờ: đỏ = ngoài ROM, cam = đột biến /
quá nhanh, vàng = dính biên ROM; danh sách DOF bị gắn cờ của frame được in trong khung.
"""
import argparse
import os

import cv2
import joblib
import numpy as np
import pandas as pd

from wham_to_keypoints import DEFAULT_SMPL, JOINTS, camera_to_world, load_smpl, to_camera

# keypoint COMETH -> tiền tố cột của CSV keypoint
KP_CSV = {
    "RShoulder": "right_shoulder", "RElbow": "right_elbow", "RWrist": "right_wrist",
    "LShoulder": "left_shoulder", "LElbow": "left_elbow", "LWrist": "left_wrist",
    "RHip": "right_hip", "RKnee": "right_knee", "RAnkle": "right_ankle",
    "LHip": "left_hip", "LKnee": "left_knee", "LAnkle": "left_ankle",
}
KPS = list(KP_CSV)
BONES = [("LShoulder", "RShoulder"), ("LHip", "RHip"), ("MidHip", "Neck"),
         ("RShoulder", "RElbow"), ("RElbow", "RWrist"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
         ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("LHip", "LKnee"), ("LKnee", "LAnkle")]

# BGR
COLOR_INPUT = (235, 235, 235)
COLOR_BEFORE = (255, 160, 40)
COLOR_AFTER = (80, 200, 60)
ISSUE_COLOR = {"out_of_rom": (60, 60, 255), "spike": (0, 140, 255), "too_fast": (0, 140, 255),
               "at_limit": (0, 215, 255)}
ISSUE_ORDER = ["out_of_rom", "spike", "too_fast", "at_limit"]
ISSUE_SHORT = {"out_of_rom": "ngoai ROM", "spike": "dot bien", "too_fast": "qua nhanh", "at_limit": "dinh bien"}


def add_virtual(pts):
    """(N, 12, 3) -> dict tên -> (N, 3), thêm MidHip (giữa hông) và Neck (giữa vai)."""
    d = {k: pts[:, i] for i, k in enumerate(KPS)}
    d["MidHip"] = (d["LHip"] + d["RHip"]) / 2
    d["Neck"] = (d["LShoulder"] + d["RShoulder"]) / 2
    return d


def load_model_joints(path, frame_ids):
    """joints_*.csv của refine_q.py -> (N, 12, 3) theo thứ tự dòng của WHAM (NaN nếu thiếu) và MPJPE."""
    df = pd.read_csv(path).set_index("frame_id")
    pts = np.full((len(frame_ids), len(KPS), 3), np.nan)
    mpjpe = np.full(len(frame_ids), np.nan)
    for r, fid in enumerate(frame_ids):
        if fid in df.index:
            row = df.loc[fid]
            pts[r] = np.array([[row["{}_{}".format(k, a)] for a in "xyz"] for k in KPS])
            mpjpe[r] = row["mpjpe"]
    return pts, mpjpe


def project(points_cam, K):
    fx, fy, cx, cy = K
    z = np.maximum(points_cam[..., 2], 1e-6)
    return np.stack([fx * points_cam[..., 0] / z + cx, fy * points_cam[..., 1] / z + cy], axis=-1)


def world_to_image(pts_world, rows, R_wc, pelvis_cam, pelvis_world, floor, K):
    p = pts_world.copy()
    p[:, :, 1] += floor[rows, None]
    cam = to_camera(p, pelvis_cam[rows], R_wc[rows], pelvis_world[rows])
    return project(cam, K)


def draw_skeleton(img, pts, color, thickness, joint_radius):
    if pts is None or np.isnan(pts["MidHip"]).any():
        return
    for a, b in BONES:
        cv2.line(img, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])), color, thickness, cv2.LINE_AA)
    for k in KPS:
        cv2.circle(img, tuple(np.int32(pts[k])), joint_radius, color, -1, cv2.LINE_AA)


def draw_flags(img, pts, flags, radius, thickness):
    if pts is None or np.isnan(pts["MidHip"]).any():
        return
    worst = {}
    for f in flags:
        kp = f["keypoint"]
        if kp not in worst or ISSUE_ORDER.index(f["issue"]) < ISSUE_ORDER.index(worst[kp]):
            worst[kp] = f["issue"]
    for kp, issue in worst.items():
        cv2.circle(img, tuple(np.int32(pts[kp])), radius, ISSUE_COLOR[issue], thickness, cv2.LINE_AA)


def draw_text_block(img, lines, x=10, y=10, scale=0.5, pad=6):
    """lines: danh sách (chữ, màu). Vẽ trên một khung tối bán trong suốt."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h = int(22 * scale / 0.5)
    w = max([cv2.getTextSize(t, font, scale, 1)[0][0] for t, _ in lines] + [0])
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w + 2 * pad, y + h * len(lines) + pad), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, (t, c) in enumerate(lines):
        cv2.putText(img, t, (x + pad, y + h * (i + 1)), font, scale, c, 1, cv2.LINE_AA)


def flag_lines(flags, value_key, max_lines):
    best = {}
    for f in flags:  # mỗi DOF một dòng, lấy lỗi nặng nhất
        if f["dof"] not in best or ISSUE_ORDER.index(f["issue"]) < ISSUE_ORDER.index(best[f["dof"]]["issue"]):
            best[f["dof"]] = f
    items = sorted(best.values(), key=lambda f: ISSUE_ORDER.index(f["issue"]))
    lines = ["{} {:.1f} [{:g},{:g}] {}".format(f["dof"], f[value_key], f["lower"], f["upper"],
                                               ISSUE_SHORT[f["issue"]]) for f in items[:max_lines]]
    out = [(l, ISSUE_COLOR[f["issue"]]) for l, f in zip(lines, items)]
    if len(items) > max_lines:
        out.append(("... +{} DOF".format(len(items) - max_lines), (200, 200, 200)))
    return out


def crop_windows(centers, size, img_w, img_h):
    """Khung cắt vuông theo từng frame (x0, y0, cạnh), bám theo khung chậu đã làm mượt."""
    side = int(min(size, img_h, img_w))
    c = pd.DataFrame(centers).interpolate(limit_direction="both").rolling(15, center=True, min_periods=1).mean()
    c = c.values
    x0 = np.clip(c[:, 0] - side / 2, 0, img_w - side).astype(int)
    y0 = np.clip(c[:, 1] - side / 2, 0, img_h - side).astype(int)
    return x0, y0, side


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wham_pkl")
    parser.add_argument("--video", required=True, help="video gốc (video đã đưa vào WHAM)")
    parser.add_argument("--smpl-model", default=DEFAULT_SMPL, help="SMPL_NEUTRAL.pkl, giống lúc chạy wham_to_keypoints.py")
    parser.add_argument("--refined", required=True, help="thư mục output của refine_q.py")
    parser.add_argument("--calib", default=None, help="calib.txt của WHAM (mặc định: cạnh file pkl)")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--ground", default="sequence", choices=["sequence", "frame", "none"])
    parser.add_argument("--layout", default="side", choices=["side", "overlay"])
    parser.add_argument("--crop", type=float, default=1.6,
                        help="cắt quanh người, cạnh = CROP x chiều cao người tính bằng pixel (0: giữ cả khung hình)")
    parser.add_argument("--height", type=int, default=720, help="chiều cao output của một khung [px]")
    parser.add_argument("--max-mpjpe", type=float, default=50.0, help="[mm] MPJPE vượt mức này hiện màu tím")
    parser.add_argument("--max-lines", type=int, default=10, help="số DOF bị gắn cờ liệt kê trong mỗi khung")
    parser.add_argument("--no-input", action="store_true", help="không vẽ skeleton input của WHAM")
    parser.add_argument("-o", "--output", default=None, help="video output (mặc định: <refined>/compare.mp4)")
    args = parser.parse_args()

    results = joblib.load(args.wham_pkl)
    subject = args.subject if args.subject is not None else max(results, key=lambda k: len(results[k]["frame_ids"]))
    d = results[subject]
    frame_ids = np.asarray(d["frame_ids"])
    row_of = {int(f): r for r, f in enumerate(frame_ids)}
    calib = args.calib or os.path.join(os.path.dirname(os.path.abspath(args.wham_pkl)), "calib.txt")
    K = np.loadtxt(calib).reshape(-1)[:4]

    J, _, _ = load_smpl(args.smpl_model)
    joints_cam, R_wc, pelvis_cam, pelvis_world, floor = camera_to_world(d, J, args.ground)
    rows = np.arange(len(frame_ids))
    uv_input = add_virtual(project(joints_cam[:, [JOINTS[KP_CSV[k]] for k in KPS]], K))
    model = {}
    for stage, fname in [("before", "joints_before.csv"), ("after", "joints_refined.csv")]:
        pts, mpjpe = load_model_joints(os.path.join(args.refined, fname), frame_ids)
        model[stage] = (add_virtual(world_to_image(pts, rows, R_wc, pelvis_cam, pelvis_world, floor, K)), mpjpe)

    flags = {"before": {}, "after": {}}
    for f in pd.read_csv(os.path.join(args.refined, "flags.csv")).to_dict("records"):
        flags[f["stage"]].setdefault(int(f["frame_id"]), []).append(f)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w, img_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n_frames <= frame_ids.max():
        print("Warning: the video has {} frames, WHAM frame ids go up to {}".format(n_frames, frame_ids.max()))

    # Khung cắt bám theo người
    if args.crop > 0:
        allpts = np.stack([uv_input[k] for k in KPS], axis=1)  # (N, 12, 2)
        body_h = np.nanmax(allpts[:, :, 1].max(axis=1) - allpts[:, :, 1].min(axis=1))
        centers = np.full((n_frames, 2), np.nan)
        valid = frame_ids < n_frames
        centers[frame_ids[valid]] = uv_input["MidHip"][valid]
        x0, y0, side = crop_windows(centers, args.crop * body_h * 1.25, img_w, img_h)
    else:
        x0, y0, side = np.zeros(n_frames, int), np.zeros(n_frames, int), None
    panel_w = args.height if side else int(round(img_w * args.height / img_h))
    zoom = args.height / (side if side else img_h)
    thick = max(2, int(round(3 / zoom)))
    radius = max(3, int(round(5 / zoom)))
    ring = max(8, int(round(16 / zoom)))

    output = args.output or os.path.join(args.refined, "compare.mp4")
    out_w = panel_w * (2 if args.layout == "side" else 1)
    writer = None
    for codec in ["avc1", "mp4v"]:
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*codec), fps, (out_w, args.height))
        if writer.isOpened():
            break
    if not writer.isOpened():
        raise RuntimeError("cannot open a video writer for " + output)

    def at(stage_pts, r):
        return None if r is None else {k: v[r] for k, v in stage_pts.items()}

    def render(frame, fid, r, stages, title):
        img = frame.copy()
        if r is not None and not args.no_input:
            draw_skeleton(img, at(uv_input, r), COLOR_INPUT, max(1, thick // 2), max(2, radius // 2))
        for stage in stages:
            color = COLOR_BEFORE if stage == "before" else COLOR_AFTER
            draw_skeleton(img, at(model[stage][0], r), color, thick, radius)
        draw_flags(img, at(model[stages[-1]][0], r) if len(stages) == 1 else at(model["before"][0], r),
                   flags[stages[0]].get(fid, []), ring, thick)
        if side:
            img = img[y0[fid]:y0[fid] + side, x0[fid]:x0[fid] + side]
        img = cv2.resize(img, (panel_w, args.height), interpolation=cv2.INTER_AREA)
        lines = [(title, COLOR_BEFORE if stages[0] == "before" and len(stages) == 1 else
                  COLOR_AFTER if len(stages) == 1 else (255, 255, 255)),
                 ("frame {}  t={:.2f}s".format(fid, fid / fps), (220, 220, 220))]
        if r is not None:
            for stage in stages:
                m = model[stage][1][r] * 1000
                lines.append(("MPJPE {} {:.0f} mm".format(stage, m),
                              (220, 120, 200) if m > args.max_mpjpe else (220, 220, 220)))
            value_key = "value_deg" if stages[0] == "before" else "refined_deg"
            lines += flag_lines(flags[stages[0]].get(fid, []), value_key, args.max_lines)
        else:
            lines.append(("(no WHAM detection)", (160, 160, 160)))
        draw_text_block(img, lines, scale=0.5 * args.height / 720)
        return img

    for fid in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        r = row_of.get(fid)
        if args.layout == "side":
            out = np.hstack([render(frame, fid, r, ["before"], "TRUOC KHI SUA"),
                             render(frame, fid, r, ["after"], "SAU KHI SUA")])
        else:
            out = render(frame, fid, r, ["before", "after"], "xanh duong: truoc  |  xanh la: sau")
        writer.write(out)
    writer.release()
    cap.release()
    print("Saved", output)


if __name__ == "__main__":
    main()
