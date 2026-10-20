"""
Kiểm tra góc khớp của kết quả IK COMETH (q.csv của run_cometh.py) so với biên độ khớp (ROM), đánh dấu các giá
trị không hợp lý, sửa chúng rồi tính lại skeleton của mô hình trước / sau khi sửa.

Cần môi trường COMETH (nimblephysics -> Linux / WSL):
    python video_pipeline/wham/refine_q.py video_pipeline/wham/out/x/ik_exact \
        --keypoints video_pipeline/wham/out/x/wham_keypoints.csv [--rom my_rom.csv]

Các kiểm tra (trên các DOF khớp, không xét 6 DOF gốc):
    out_of_rom   giá trị nằm ngoài [lower, upper] của bảng ROM
    at_limit     giá trị dính sát biên ROM: IK muốn đi xa hơn, keypoint không khớp với mô hình
                 (không xét các DOF trong --no-limit-check: COMETH cố ý khoá cột sống / xương bả vai trong vài độ)
    spike        lệch riêng lẻ so với median 5 frame lớn hơn --spike-deg
    too_fast     vận tốc góc vượt --max-vel [độ/s]
    bad_fit      (cả frame) MPJPE của mô hình sau khi khớp lớn hơn --max-mpjpe [mm]

Cách sửa (hậu xử lý đơn giản trên q, không chạy lại IK):
    1. bỏ các mẫu spike / too_fast rồi nội suy lại (PCHIP) nếu đoạn lỗi dài <= --max-gap frame
    2. lọc thông thấp Butterworth hai chiều (không trễ pha) ở --cutoff Hz trên các DOF khớp
       (và trên DOF gốc nếu bật --filter-root)
    3. kẹp vào ROM
'at_limit' và 'bad_fit' không sửa được bằng cách này, chỉ được báo. Làm mượt từng DOF riêng lẻ khiến mô hình
lệch keypoint thêm một chút (so sánh MPJPE trước / sau).

Output (trong --out, mặc định <ik_dir>/refined):
    q_refined.csv                   toạ độ khớp đã sửa (cùng format với q.csv)
    flags.csv                       mỗi dòng là một (frame, DOF, lỗi), trước và sau khi sửa
    joints_before.csv, joints_refined.csv
                                    12 keypoint của mô hình trong hệ của CSV input (Y-up, mét) và MPJPE,
                                    dùng cho overlay_video.py
    rom_used.csv                    bảng ROM đã dùng (sửa rồi truyền lại bằng --rom)
    report.html                     bảng mọi góc khớp từng frame, tô màu các giá trị không hợp lý
    joint_angles_refined.png        trước / sau của các DOF bị đánh dấu nhiều nhất
"""
import argparse
import html
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run_cometh.py
from run_cometh import RX, build_skeleton, get_times, load_targets  # noqa: E402
from COMETH import parameters as COMETH_parameters  # noqa: E402

ISSUES = ["out_of_rom", "at_limit", "spike", "too_fast"]
N_ROOT = 6

# ROM [độ] cho các DOF mà COMETH/parameters.py để ở +-180
EXTRA_ROM = {
    "subtalar_angle": (-20, 20),
    "mtp_angle": (-30, 60),
    "head_bending": (-40, 40),
    "head_extension": (-50, 60),
    "head_twist": (-70, 70),
    "pro_sup": (-90, 90),
    "wrist_flexion": (-70, 80),
    "wrist_deviation": (-25, 35),
}


def dof_keypoint(name):
    """Keypoint dùng để vẽ lỗi của DOF này (MidHip / Neck do overlay_video.py tính)."""
    side = "R" if "_r_" in name + "_" else "L" if "_l_" in name + "_" else ""
    for prefix, kp in [("hip", "Hip"), ("knee", "Knee"), ("ankle", "Ankle"), ("subtalar", "Ankle"),
                       ("mtp", "Ankle"), ("scapula", "Shoulder"), ("shoulder", "Shoulder"),
                       ("elbow", "Elbow"), ("pro_sup", "Elbow"), ("wrist", "Wrist")]:
        if side and name.startswith(prefix):
            return side + kp
    return "MidHip" if name.startswith(("lumbar", "pelvis")) else "Neck"


def default_rom(names):
    lo = COMETH_parameters.Q_LOWER_BOUND.astype(float).copy()
    hi = COMETH_parameters.Q_UPPER_BOUND.astype(float).copy()
    for i, n in enumerate(names):
        if lo[i] == -180 and hi[i] == 180:
            for prefix, (a, b) in EXTRA_ROM.items():
                if n.startswith(prefix):
                    lo[i], hi[i] = a, b
    return pd.DataFrame({"dof": names, "lower": lo, "upper": hi})


def load_rom(path, names):
    rom = default_rom(names)
    if path:
        user = pd.read_csv(path).set_index("dof")
        for i, n in enumerate(names):
            if n in user.index:
                rom.loc[i, ["lower", "upper"]] = user.loc[n, ["lower", "upper"]].values
    return rom


def check(Qdeg, times, lo, hi, names, args):
    """Mặt nạ boolean (N, D) cho từng loại lỗi, chỉ trên các DOF khớp."""
    n, D = Qdeg.shape
    ignored = tuple(p for p in args.no_limit_check.split(",") if p)
    limit_checked = np.array([not n_.startswith(ignored) for n_ in names[N_ROOT:]]) if ignored else True
    flags = {k: np.zeros((n, D), bool) for k in ISSUES}
    J = slice(N_ROOT, D)
    q = Qdeg[:, J]
    moving = np.ptp(q, axis=0) > 1e-6  # DOF bị mô hình khoá cứng thì không bao giờ bị đánh dấu
    tol = args.limit_tol
    flags["out_of_rom"][:, J] = ((q < lo[J] - tol) | (q > hi[J] + tol)) & moving
    flags["at_limit"][:, J] = ((np.abs(q - lo[J]) <= tol) | (np.abs(q - hi[J]) <= tol)) & moving & limit_checked
    if n >= 5:
        flags["spike"][:, J] = np.abs(q - median_filter(q, size=(5, 1), mode="nearest")) > args.spike_deg
    if n >= 2:
        v = np.abs(np.diff(q, axis=0)) / np.diff(times)[:, None]
        fast = v > args.max_vel
        flags["too_fast"][1:, J] |= fast
        flags["too_fast"][:-1, J] |= fast
    return flags


def interpolate_gaps(x, t, bad, max_gap):
    """Thay các đoạn mẫu lỗi (dài <= max_gap, có mẫu tốt ở hai bên) bằng nội suy PCHIP."""
    x = x.copy()
    good = ~bad
    if good.sum() < 2:
        return x
    f = PchipInterpolator(t[good], x[good])
    i = 0
    while i < len(x):
        if bad[i]:
            j = i
            while j < len(x) and bad[j]:
                j += 1
            if j - i <= max_gap and i > 0 and j < len(x):
                x[i:j] = f(t[i:j])
            i = j
        else:
            i += 1
    return x


def repair(Q, times, flags, lo_rad, hi_rad, args):
    Qr = Q.copy()
    Qr[:, :3] = np.unwrap(Qr[:, :3], axis=0)
    bad = flags["spike"] | flags["too_fast"]
    for d in range(N_ROOT, Q.shape[1]):
        if bad[:, d].any():
            Qr[:, d] = interpolate_gaps(Qr[:, d], times, bad[:, d], args.max_gap)
    fs = 1.0 / np.median(np.diff(times))
    if args.cutoff > 0 and len(Qr) > 15:
        b, a = butter(4, min(args.cutoff / (fs / 2), 0.99))
        first = 0 if args.filter_root else N_ROOT
        Qr[:, first:] = filtfilt(b, a, Qr[:, first:], axis=0)
    if not args.filter_root:
        Qr[:, :N_ROOT] = Q[:, :N_ROOT]
    Qr[:, N_ROOT:] = np.clip(Qr[:, N_ROOT:], lo_rad[N_ROOT:], hi_rad[N_ROOT:])
    return Qr


def forward(s, Q):
    out = []
    for q in Q:
        s._nimble.setPositions(q)
        out.append(s.to_numpy())
    return np.array(out)


def flags_table(flags, Qdeg, Qref_deg, times, frame_ids, names, rom, stage):
    rows = []
    for issue in ISSUES:
        for i, d in zip(*np.nonzero(flags[issue])):
            rows.append({"stage": stage, "frame": i, "frame_id": frame_ids[i], "time": times[i],
                         "dof": names[d], "keypoint": dof_keypoint(names[d]), "issue": issue,
                         "value_deg": Qdeg[i, d], "refined_deg": Qref_deg[i, d],
                         "lower": rom.lower[d], "upper": rom.upper[d]})
    return rows


def save_joints(path, s, fitted, target, times, frame_ids):
    """Đưa keypoint của mô hình về hệ của CSV input (Z-up -> Y-up)."""
    n = len(fitted)
    pts = fitted.reshape(n, -1, 3) @ RX  # RX trực giao: nghịch đảo = chuyển vị
    df = pd.DataFrame({"time": times[:n], "frame_id": frame_ids[:n]})
    for k, kp in enumerate(s.kps):
        for a, axis in enumerate("xyz"):
            df["{}_{}".format(kp, axis)] = pts[:, k, a]
    mpjpe = np.linalg.norm(fitted.reshape(n, -1, 3) - target[:n].reshape(n, -1, 3), axis=2).mean(axis=1)
    df["mpjpe"] = mpjpe
    df.to_csv(path, index=False)
    return mpjpe


CSS = """
body{font-family:system-ui,sans-serif;margin:16px;color:#1d1d1f;background:#fff}
h1{font-size:20px}h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;font-size:12px}
th,td{border:1px solid #ddd;padding:2px 5px;text-align:right;white-space:nowrap}
th{background:#f4f4f6;position:sticky;top:0;z-index:1}
td.f{position:sticky;left:0;background:#f9f9fb;text-align:left}
.wrap{max-height:75vh;overflow:auto;border:1px solid #ccc}
.out_of_rom{background:#ff6b6b;color:#fff}
.at_limit{background:#ffe08a}
.spike,.too_fast{background:#ffa94d}
.bad_fit{background:#d0bfff}
.legend span{display:inline-block;padding:2px 8px;margin-right:6px;border:1px solid #ccc}
.controls{margin:10px 0}
tr.clean.hide{display:none}
"""

JS = """
function setMode(m){document.querySelectorAll('td[data-o]').forEach(function(c){
  c.textContent=m==='o'?c.dataset.o:c.dataset.r;
  c.className=m==='o'?c.dataset.co:c.dataset.cr;});}
function onlyBad(on){document.querySelectorAll('tr.clean').forEach(function(r){r.classList.toggle('hide',on);});}
"""


def cell_class(flags, i, d):
    # lỗi nặng nhất được ưu tiên
    for issue in ["out_of_rom", "spike", "too_fast", "at_limit"]:
        if flags[issue][i, d]:
            return issue
    return ""


def write_report(path, names, rom, Qdeg, Qref_deg, flags, flags_ref, mpjpe, mpjpe_ref, times, frame_ids, args):
    D = len(names)
    cols = [d for d in range(N_ROOT, D) if np.ptp(Qdeg[:, d]) > 1e-6 or np.ptp(Qref_deg[:, d]) > 1e-6]
    bad_fit = mpjpe * 1000 > args.max_mpjpe
    bad_fit_ref = mpjpe_ref * 1000 > args.max_mpjpe
    e = html.escape

    summary = ["<tr><th style='text-align:left'>DOF</th><th>ROM [deg]</th><th>min</th><th>max</th>"]
    summary += ["<th>{} (before → after)</th>".format(k) for k in ISSUES]
    summary.append("</tr>")
    for d in cols:
        counts = ["<td class='{}'>{} → {}</td>".format(k if flags[k][:, d].any() else "", flags[k][:, d].sum(),
                                                       flags_ref[k][:, d].sum()) for k in ISSUES]
        summary.append("<tr><td class='f'>{}</td><td>[{:g}, {:g}]</td><td>{:.1f}</td><td>{:.1f}</td>{}</tr>".format(
            e(names[d]), rom.lower[d], rom.upper[d], Qdeg[:, d].min(), Qdeg[:, d].max(), "".join(counts)))

    rows = ["<tr><th>frame</th><th>time [s]</th><th>MPJPE [mm]</th>" +
            "".join("<th>{}<br><small>[{:g}, {:g}]</small></th>".format(e(names[d]), rom.lower[d], rom.upper[d])
                    for d in cols) + "</tr>"]
    for i in range(len(Qdeg)):
        cells, dirty = [], bad_fit[i] or bad_fit_ref[i]
        for d in cols:
            co, cr = cell_class(flags, i, d), cell_class(flags_ref, i, d)
            dirty |= bool(co or cr)
            tip = "ROM [{:g}, {:g}] | before {:.1f} {} | after {:.1f} {}".format(
                rom.lower[d], rom.upper[d], Qdeg[i, d], co, Qref_deg[i, d], cr)
            cells.append("<td class='{co}' data-co='{co}' data-cr='{cr}' data-o='{o:.1f}' data-r='{r:.1f}' "
                         "title='{tip}'>{o:.1f}</td>".format(co=co, cr=cr, o=Qdeg[i, d], r=Qref_deg[i, d], tip=tip))
        fit = "<td class='{c}' data-co='{c}' data-cr='{cr}' data-o='{o:.0f}' data-r='{r:.0f}'>{o:.0f}</td>".format(
            c="bad_fit" if bad_fit[i] else "", cr="bad_fit" if bad_fit_ref[i] else "",
            o=mpjpe[i] * 1000, r=mpjpe_ref[i] * 1000)
        rows.append("<tr class='{}'><td class='f'>{} <small>(#{})</small></td><td>{:.3f}</td>{}{}</tr>".format(
            "" if dirty else "clean", i, frame_ids[i], times[i], fit, "".join(cells)))

    n_before = sum(int(flags[k][:, cols].sum()) for k in ISSUES)
    n_after = sum(int(flags_ref[k][:, cols].sum()) for k in ISSUES)
    doc = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Joint angle check</title>
<style>{css}</style><script>{js}</script></head><body>
<h1>Kiểm tra góc khớp</h1>
<p>{n} frame · {nd} DOF · ô bị đánh dấu: <b>{nb}</b> trước khi sửa → <b>{na}</b> sau khi sửa.
MPJPE trung bình: {m0:.1f} mm → {m1:.1f} mm. Đơn vị góc: độ.</p>
<div class="legend"><span class="out_of_rom">ngoài ROM</span><span class="spike">nhảy đột ngột / quá nhanh
(&gt; {spike:g}° lệch median hoặc &gt; {vel:g}°/s)</span><span class="at_limit">dính biên ROM (± {tol:g}°)</span>
<span class="bad_fit">MPJPE &gt; {fit:g} mm</span></div>
<h2>Tổng hợp theo DOF</h2><div class="wrap"><table>{summary}</table></div>
<h2>Từng frame</h2>
<div class="controls">Giá trị hiển thị:
<label><input type="radio" name="m" checked onchange="setMode('o')"> trước khi sửa</label>
<label><input type="radio" name="m" onchange="setMode('r')"> sau khi sửa</label>
&nbsp; <label><input type="checkbox" onchange="onlyBad(this.checked)"> chỉ hiện frame có lỗi</label>
&nbsp; <small>Rê chuột lên ô để xem cả hai giá trị.</small></div>
<div class="wrap"><table>{rows}</table></div></body></html>""".format(
        css=CSS, js=JS, n=len(Qdeg), nd=len(cols), nb=n_before, na=n_after,
        m0=1000 * mpjpe.mean(), m1=1000 * mpjpe_ref.mean(), spike=args.spike_deg, vel=args.max_vel,
        tol=args.limit_tol, fit=args.max_mpjpe, summary="".join(summary), rows="".join(rows))
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def plot(path, names, rom, times, Qdeg, Qref_deg, flags, n_plots=6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    score = sum(flags[k][:, N_ROOT:].sum(axis=0) for k in ISSUES)
    order = [N_ROOT + d for d in np.argsort(-score)[:n_plots] if score[d] > 0]
    if not order:
        return
    fig, axes = plt.subplots(len(order), 1, figsize=(10, 2.2 * len(order)), sharex=True, squeeze=False)
    for ax, d in zip(axes[:, 0], order):
        ax.axhspan(rom.lower[d], rom.upper[d], color="tab:green", alpha=0.08)
        ax.plot(times, Qdeg[:, d], color="tab:red", lw=1, label="before")
        ax.plot(times, Qref_deg[:, d], color="tab:green", lw=1.5, label="after")
        for k, c in [("spike", "tab:orange"), ("too_fast", "tab:orange"), ("at_limit", "gold"),
                     ("out_of_rom", "red")]:
            m = flags[k][:, d]
            ax.scatter(times[m], Qdeg[m, d], s=10, color=c, zorder=3)
        lo, hi = Qdeg[:, d].min(), Qdeg[:, d].max()
        ax.set_ylim(lo - 0.15 * (hi - lo + 1), hi + 0.15 * (hi - lo + 1))
        ax.set_ylabel(names[d], fontsize=8)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ik_dir", help="thư mục output của run_cometh.py --mode ik (có q.csv)")
    parser.add_argument("--keypoints", required=True, help="CSV keypoint đã đưa vào run_cometh.py")
    parser.add_argument("--out", default=None, help="thư mục output (mặc định: <ik_dir>/refined)")
    parser.add_argument("--rom", default=None, help="CSV dof,lower,upper [độ] thay cho ROM mặc định")
    parser.add_argument("--limit-tol", type=float, default=0.5, help="[độ] khoảng cách tới biên tính là at_limit")
    parser.add_argument("--spike-deg", type=float, default=10.0, help="[độ] độ lệch so với median 5 frame")
    parser.add_argument("--max-vel", type=float, default=1000.0, help="[độ/s] vận tốc góc khớp tối đa")
    parser.add_argument("--max-mpjpe", type=float, default=50.0, help="[mm] frame vượt mức này bị đánh dấu bad_fit")
    parser.add_argument("--max-gap", type=int, default=10, help="[frame] đoạn mẫu lỗi dài nhất được nội suy lại")
    parser.add_argument("--cutoff", type=float, default=12.0, help="[Hz] tần số cắt lọc thông thấp, 0 để tắt")
    parser.add_argument("--filter-root", action="store_true", help="lọc thông thấp cả 6 DOF gốc")
    parser.add_argument("--no-limit-check", default="lumbar_,thorax_,scapula_",
                        help="các tiền tố DOF (cách nhau bởi dấu phẩy) không xét at_limit ('' để xét tất cả)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.ik_dir, "refined")
    os.makedirs(out_dir, exist_ok=True)

    df_q = pd.read_csv(os.path.join(args.ik_dir, "q.csv"))
    times = df_q.pop("time").values
    names = list(df_q.columns)
    Q = df_q.values
    n = len(Q)

    markers = pd.read_csv(args.keypoints).iloc[:n].reset_index(drop=True)
    target = load_targets(markers)
    frame_ids = markers["frame_id"].values if "frame_id" in markers else np.arange(n)
    if not np.allclose(get_times(markers)[:n], times, atol=1e-6):
        print("Warning: the times of --keypoints do not match q.csv")

    rom = load_rom(args.rom, names)
    rom.to_csv(os.path.join(out_dir, "rom_used.csv"), index=False)
    lo, hi = rom.lower.values, rom.upper.values

    Qdeg = np.degrees(Q)
    flags = check(Qdeg, times, lo, hi, names, args)
    Qr = repair(Q, times, flags, np.radians(lo), np.radians(hi), args)
    Qref_deg = np.degrees(Qr)
    flags_ref = check(Qref_deg, times, lo, hi, names, args)

    out_q = pd.DataFrame(Qr, columns=names)
    out_q.insert(0, "time", times)
    out_q.to_csv(os.path.join(out_dir, "q_refined.csv"), index=False)

    rows = flags_table(flags, Qdeg, Qref_deg, times, frame_ids, names, rom, "before")
    rows += flags_table(flags_ref, Qdeg, Qref_deg, times, frame_ids, names, rom, "after")
    pd.DataFrame(rows, columns=["stage", "frame", "frame_id", "time", "dof", "keypoint", "issue", "value_deg",
                                "refined_deg", "lower", "upper"]).to_csv(os.path.join(out_dir, "flags.csv"),
                                                                          index=False)

    print("Rebuilding the scaled BSM model for forward kinematics...")
    s, _ = build_skeleton(target[0])
    mpjpe = save_joints(os.path.join(out_dir, "joints_before.csv"), s, forward(s, Q), target, times, frame_ids)
    mpjpe_ref = save_joints(os.path.join(out_dir, "joints_refined.csv"), s, forward(s, Qr), target, times,
                            frame_ids)

    write_report(os.path.join(out_dir, "report.html"), names, rom, Qdeg, Qref_deg, flags, flags_ref,
                 mpjpe, mpjpe_ref, times, frame_ids, args)
    plot(os.path.join(out_dir, "joint_angles_refined.png"), names, rom, times, Qdeg, Qref_deg, flags)

    print("{:<12} {:>8} {:>8}".format("issue", "before", "after"))
    for k in ISSUES:
        print("{:<12} {:>8} {:>8}".format(k, int(flags[k].sum()), int(flags_ref[k].sum())))
    print("{:<12} {:>8} {:>8}".format("bad_fit", int((mpjpe * 1000 > args.max_mpjpe).sum()),
                                      int((mpjpe_ref * 1000 > args.max_mpjpe).sum())))
    print("MPJPE w.r.t. the input keypoints: {:.1f} mm -> {:.1f} mm".format(1000 * mpjpe.mean(),
                                                                          1000 * mpjpe_ref.mean()))
    print("Results saved in", out_dir)


if __name__ == "__main__":
    main()
