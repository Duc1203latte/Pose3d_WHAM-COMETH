"""
Chạy COMETH trên một CSV keypoint mà không sửa code gốc. Dùng chung cho cả hai nhánh:
wham/wham_to_keypoints.py, mediapipe/video_to_keypoints.py và inverse_dynamics/data.csv đều có format này.

Cần môi trường COMETH (nimblephysics -> Linux / WSL):
    python video_pipeline/run_cometh.py video_pipeline/wham/out/x/wham_keypoints.csv \
        --out video_pipeline/wham/out/x/ik_exact --ik exact
    python video_pipeline/run_cometh.py video_pipeline/mediapipe/out/my_video.csv --out video_pipeline/mediapipe/out/my_video_ik
    python video_pipeline/run_cometh.py my.csv --out out_dyn --mode dynamics
    python video_pipeline/run_cometh.py my.csv --out out_ik --gui   # http://localhost:8080

Chế độ:
    ik        Khớp mô hình BSM vào keypoint từng frame.
              --ik qp    : DynamicSkeleton.filter (qpIK + Kalman cho từng DOF), tracker của COMETH
              --ik exact : DynamicSkeleton.exact_scale(to_scale=False), như trong test_dynamics.ipynb
    dynamics  Chạy nguyên inverse_dynamics/test_dynamics_claude.py, chỉ chuyển hướng đường dẫn CSV hardcode
              sang file được truyền vào (đã resample về dt=0.033 s cố định của script), rồi lưu kết quả.

Output (trong --out):
    q.csv          toạ độ khớp của mô hình (49 DOF, radian / mét) theo từng frame
    keypoints.csv  12 keypoint của mô hình sau khi khớp (Z-up) và MPJPE từng frame so với input
    tau.csv, grf.csv   (chỉ dynamics) mô-men khớp [Nm] và lực phản lực đất theo phương đứng [N]
    *.png          biểu đồ nhanh
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from COMETH import Skeleton, DynamicSkeleton  # noqa: E402

DYNAMICS_SCRIPT = os.path.join(ROOT, "inverse_dynamics", "test_dynamics_claude.py")
DYNAMICS_DT = 0.033  # hardcode trong test_dynamics_claude.py

# Cùng bảng ánh xạ và phép xoay Y-up -> Z-up như các script trong inverse_dynamics/
MARKERS_DICT = {
    'RKnee': 'right_knee',
    'LWrist': 'left_wrist',
    'RHip': 'right_hip',
    'RShoulder': 'right_shoulder',
    'LElbow': 'left_elbow',
    'LHip': 'left_hip',
    'RElbow': 'right_elbow',
    'RWrist': 'right_wrist',
    'LKnee': 'left_knee',
    'LShoulder': 'left_shoulder',
    'RAnkle': 'right_ankle',
    'LAnkle': 'left_ankle'
}
RX = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])


def load_targets(markers):
    """Dataframe CSV -> (N, 36) keypoint theo thứ tự của COMETH (s.kps), Z-up."""
    cols = []
    for name in MARKERS_DICT.values():
        cols += [name + "_x", name + "_y", name + "_z"]
    pts = markers[cols].values.reshape(len(markers), -1, 3)
    return (pts @ RX.T).reshape(len(markers), -1)


def get_times(markers):
    if "timestamp" in markers.columns:
        return (markers["timestamp"].values - markers["timestamp"].values[0]) / 1e9
    return np.arange(len(markers)) * DYNAMICS_DT


def resample(markers, dt):
    """Resample tuyến tính mọi cột toạ độ về bước thời gian cố định dt [s]."""
    t = get_times(markers)
    t_new = np.arange(0, t[-1] + 1e-9, dt)
    out = pd.DataFrame({"timestamp": np.round(t_new * 1e9).astype(np.int64),
                        "frame_id": np.arange(len(t_new)), "body_id": 0})
    for c in markers.columns:
        if c.endswith(("_x", "_y", "_z")):
            out[c] = np.interp(t_new, t, markers[c].values)
    return out


def dof_names(s):
    return [s._nimble.getDofByIndex(i).getName() for i in range(s._nimble.getNumDofs())]


def build_skeleton(target0, vel_scale=1.0):
    s12 = Skeleton()
    s = DynamicSkeleton()
    s.hip_correction = False
    s.qdot_l = s.qdot_l * vel_scale
    s.qdot_u = s.qdot_u * vel_scale
    s12.load_from_numpy(target0.reshape(-1, 3), s.kps)
    s.load_from_BODY12(s12)
    s.exact_scale()
    return s, s12


def run_ik(target, times, method, vel_scale, max_frames):
    s, s12 = build_skeleton(target[0], vel_scale)
    n = min(len(target), max_frames) if max_frames else len(target)
    Q, fitted, mpjpe = [], [], []
    t0 = time.time()
    for i in range(n):
        kp = target[i].reshape(-1, 3)
        if method == "qp":
            dt = times[i] - times[i - 1] if i > 0 else 1 / 30
            s.filter([kp], dt=dt)
        else:
            s12.load_from_numpy(kp, s.kps)
            s.load_from_BODY12(s12)
            s.exact_scale(to_scale=False, max_iterations=100)
        Q.append(s._nimble.getPositions().copy())
        pos = s.to_numpy()
        fitted.append(pos)
        mpjpe.append(np.mean(np.linalg.norm(pos.reshape(-1, 3) - kp, axis=1)))
        if (i + 1) % 20 == 0:
            print("  frame {}/{}  MPJPE {:.3f} m  ({:.2f} s/frame)".format(
                i + 1, n, np.mean(mpjpe), (time.time() - t0) / (i + 1)))
    return s, np.array(Q), np.array(fitted), np.array(mpjpe)


def run_dynamics(csv_path):
    """Chạy nguyên test_dynamics_claude.py, cho nó đọc csv_path thay vì data.csv hardcode."""
    original_read_csv = pd.read_csv

    def redirected_read_csv(path, *args, **kwargs):
        if isinstance(path, str) and os.path.basename(path) == "data.csv":
            path = csv_path
        return original_read_csv(path, *args, **kwargs)

    g = {"__name__": "__main__", "__file__": DYNAMICS_SCRIPT}
    pd.read_csv = redirected_read_csv
    try:
        with open(DYNAMICS_SCRIPT) as f:
            exec(compile(f.read(), DYNAMICS_SCRIPT, "exec"), g)
    except Exception as e:  # giữ lại các frame đã tính được trước khi lỗi
        print("test_dynamics_claude.py stopped with {}: {}".format(type(e).__name__, e))
    finally:
        pd.read_csv = original_read_csv
    return g


def save_outputs(out_dir, s, times, Q, fitted, mpjpe, target):
    names = dof_names(s)
    n = len(Q)
    df_q = pd.DataFrame(Q, columns=names)
    df_q.insert(0, "time", times[:n])
    df_q.to_csv(os.path.join(out_dir, "q.csv"), index=False)

    cols = ["{}_{}".format(k, a) for k in s.kps for a in "xyz"]
    df_kp = pd.DataFrame(fitted, columns=cols)
    df_kp.insert(0, "time", times[:n])
    df_kp["mpjpe"] = mpjpe
    df_kp.to_csv(os.path.join(out_dir, "keypoints.csv"), index=False)
    print("Frames: {}  |  mean MPJPE w.r.t. the input keypoints: {:.1f} mm".format(n, 1000 * np.mean(mpjpe)))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l",
            "elbow_flexion_r", "elbow_flexion_l"]
    plot = [p for p in plot if p in names]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for p in plot:
        axes[0].plot(times[:n], np.degrees(df_q[p]), label=p)
    axes[0].set_ylabel("angle [deg]")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(times[:n], 1000 * np.asarray(mpjpe))
    axes[1].set_ylabel("MPJPE [mm]")
    axes[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "joint_angles.png"), dpi=120)
    plt.close(fig)


def save_dynamics(out_dir, g, times):
    s, results = g["s"], g.get("results", [])
    if not results:
        print("No dynamics frame was solved.")
        return
    n = len(results)
    names = dof_names(s)[6:]
    df_tau = pd.DataFrame([r["tau"] for r in results], columns=names)
    df_tau.insert(0, "time", times[:n])
    df_tau.to_csv(os.path.join(out_dir, "tau.csv"), index=False)

    weight = s._nimble.getMass() * 9.81
    fz = np.array([np.sum(r["fc"][2::3]) for r in results])
    df_grf = pd.DataFrame({"time": times[:n],
                           "n_contacts": [len(r["contact_info"]) for r in results],
                           "grf_z": fz,
                           "grf_z_over_body_weight": fz / weight})
    df_grf.to_csv(os.path.join(out_dir, "grf.csv"), index=False)
    print("Body mass {:.1f} kg  |  mean vertical GRF {:.2f} x body weight".format(
        s._nimble.getMass(), np.mean(fz) / weight))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for p in ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"]:
        if p in df_tau:
            axes[0].plot(df_tau["time"], df_tau[p], label=p)
    axes[0].set_ylabel("torque [Nm]")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(df_grf["time"], df_grf["grf_z_over_body_weight"])
    axes[1].axhline(1.0, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("vertical GRF / body weight")
    axes[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dynamics.png"), dpi=120)
    plt.close(fig)


def replay_gui(s, Q, target, dt, port=8080):
    import nimblephysics as nimble
    gui = nimble.NimbleGUI()
    gui.serve(port)
    print("Open http://localhost:{} (Ctrl+C to stop)".format(port))
    try:
        while True:
            for i in range(len(Q)):
                s._nimble.setPositions(Q[i])
                gui.nativeAPI().renderSkeleton(s._nimble)
                for j in range(target.shape[1] // 3):
                    gui.nativeAPI().createSphere(key=str(j), radii=np.array([0.02, 0.02, 0.02]),
                                                 pos=target[i, 3 * j:3 * j + 3], color=np.array([0, 255, 0, 1]))
                time.sleep(dt)
    except KeyboardInterrupt:
        gui.stopServing()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--out", default=None, help="thư mục output (mặc định: <csv>_cometh)")
    parser.add_argument("--mode", default="ik", choices=["ik", "dynamics"])
    parser.add_argument("--ik", default="qp", choices=["qp", "exact"])
    parser.add_argument("--vel-scale", type=float, default=1.0,
                        help="nhân giới hạn vận tốc khớp BSM mà qpIK dùng (chuyển động nhanh có thể cần > 1)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--gui", action="store_true", help="phát lại kết quả trong nimble GUI")
    args = parser.parse_args()

    out_dir = args.out or os.path.splitext(args.csv)[0] + "_cometh"
    os.makedirs(out_dir, exist_ok=True)
    markers = pd.read_csv(args.csv)
    if args.max_frames:
        markers = markers.iloc[:args.max_frames].reset_index(drop=True)

    if args.mode == "ik":
        target = load_targets(markers)
        times = get_times(markers)
        s, Q, fitted, mpjpe = run_ik(target, times, args.ik, args.vel_scale, args.max_frames)
    else:
        markers = resample(markers, DYNAMICS_DT)
        csv_resampled = os.path.join(out_dir, "input_resampled.csv")
        markers.to_csv(csv_resampled)
        print("Resampled to dt={} s ({} frames), running {}".format(
            DYNAMICS_DT, len(markers), os.path.relpath(DYNAMICS_SCRIPT, ROOT)))
        target = load_targets(markers)
        times = get_times(markers)
        g = run_dynamics(csv_resampled)
        s = g["s"]
        Q = np.array(g.get("Q", []))
        n = len(Q)
        if n == 0:
            print("No frame was solved by the dynamics script: nothing to save.")
            return
        # Q[i] là trạng thái trước khi cập nhật frame i: dùng trạng thái kế tiếp để so với target[i]
        states = list(Q[1:]) + ([s._nimble.getPositions().copy()] if n else [])
        fitted = []
        for q in states:
            s._nimble.setPositions(q)
            fitted.append(s.to_numpy())
        fitted = np.array(fitted)
        mpjpe = np.linalg.norm(fitted.reshape(n, -1, 3) - target[:n].reshape(n, -1, 3), axis=2).mean(axis=1)
        Q = np.array(states)
        save_dynamics(out_dir, g, times)

    save_outputs(out_dir, s, times, Q, fitted, mpjpe, target)
    print("Results saved in", out_dir)

    if args.gui:
        replay_gui(s, Q, target, float(np.median(np.diff(times))) if len(times) > 1 else DYNAMICS_DT)


if __name__ == "__main__":
    main()
