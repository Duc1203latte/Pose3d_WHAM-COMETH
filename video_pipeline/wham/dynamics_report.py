"""
Inverse dynamics trên kết quả IK của COMETH (q.csv, góc khớp KHÔNG bị sửa), và một bảng HTML đánh dấu các giá
trị không hợp lý về mặt vật lý.

Cần môi trường COMETH (nimblephysics + cvxpy -> Linux / WSL):
    python video_pipeline/wham/dynamics_report.py video_pipeline/wham/out/x/ik_exact \
        --keypoints video_pipeline/wham/out/x/wham_keypoints.csv --flight 12-48

Với mỗi frame:
    1. q được unwrap và lọc thông thấp ở --cutoff Hz (chỉ để lấy đạo hàm), dq và ddq tính bằng sai phân trung tâm
    2. tiếp xúc bàn chân. Độ cao toàn cục của dựng hình một camera không đáng tin (WHAM có thể giữ người trên mặt
       đất lúc đang nhảy rồi để người lơ lửng sau khi tiếp đất), nên các pha bay được khai báo bằng --flight
       (frame id đọc từ video). Ngoài các pha đó, bàn chân thấp hơn được coi là chạm đất, và chân còn lại cũng
       vậy nếu điểm thấp nhất của nó cao hơn chưa tới --contact-height; trên một bàn chân, các điểm gót / mũi
       chân cao hơn điểm thấp nhất của chân đó chưa tới --contact-height là điểm tiếp xúc.
    3. bình phương tối thiểu với ma trận khối lượng M, lực Coriolis + trọng lực h và Jacobian tiếp xúc Jc
       của nimblephysics:
           M ddq + h = [r ; tau] + Jc^T f,   f nằm trong hình chóp ma sát (--mu), f_z >= 0
       cực tiểu hoá lực / mô-men dư r ở khung chậu (6 DOF ảo); từ đó tau suy ra chính xác.
       r là phần không khớp và không điểm tiếp xúc nào giải thích được: phải xấp xỉ 0.

Cờ:
    torque       |tau| vượt mức sinh lý tối đa (ước lượng thô) của khớp [Nm/kg], xem TORQUE_MAX
    grf          lực phản lực đất theo phương đứng vượt --max-grf lần trọng lượng
    residual_f   lực dư vượt --max-residual-f lần trọng lượng
    residual_m   mô-men dư vượt --max-residual-m lần trọng lượng x 1 m
Output trong --out (mặc định <ik_dir>/dynamics): tau.csv, grf_residual.csv, flags_dynamics.csv, dynamics.html,
dynamics.png
"""
import argparse
import html
import json
import os
import sys

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run_cometh.py
from change_report import CSS, JS  # noqa: E402
from run_cometh import build_skeleton, dof_names, load_targets  # noqa: E402

N_ROOT = 6
G = 9.81
# Cận trên thô của mô-men khớp trong vận động thể thao [Nm/kg] (tiếp đất sau nhảy, chạy nước rút); chỉ để sàng lọc
TORQUE_MAX = [
    ("hip_flexion", 4.0), ("hip_adduction", 3.0), ("hip_rotation", 1.5), ("knee_angle", 4.0),
    ("ankle_angle", 3.5), ("subtalar_angle", 1.5), ("mtp_angle", 1.0),
    ("lumbar_twist", 1.5), ("thorax_twist", 1.5), ("lumbar", 4.0), ("thorax", 4.0), ("head", 0.6),
    ("scapula", 1.5), ("shoulder", 1.5), ("elbow_flexion", 1.0), ("pro_sup", 0.2), ("wrist", 0.3),
]
FOOT_POINTS = {  # body, độ lệch trong hệ của body (như trong inverse_dynamics/test_dynamics_claude.py)
    "L": [("calcn_l", [0.0, 0.0, -0.03]), ("calcn_l", [0.0, 0.0, 0.03]),
          ("calcn_l", [0.15, 0.0, -0.03]), ("calcn_l", [0.15, 0.0, 0.03])],
    "R": [("calcn_r", [0.0, 0.0, -0.03]), ("calcn_r", [0.0, 0.0, 0.03]),
          ("calcn_r", [0.15, 0.0, -0.03]), ("calcn_r", [0.15, 0.0, 0.03])],
}
ISSUE_VI = {"torque": "mô-men quá lớn", "grf": "lực đất quá lớn", "residual_f": "lực dư lớn",
            "residual_m": "mô-men dư lớn"}


def torque_limit(name):
    for prefix, v in TORQUE_MAX:
        if name.startswith(prefix):
            return v
    return 1.0


def differentiate(Q, t, cutoff):
    Q = Q.copy()
    Q[:, :3] = np.unwrap(Q[:, :3], axis=0)
    fs = 1.0 / np.median(np.diff(t))
    if cutoff > 0:
        b, a = butter(4, min(cutoff / (fs / 2), 0.99))
        Q = filtfilt(b, a, Q, axis=0)
    dQ = np.gradient(Q, t, axis=0)
    ddQ = np.gradient(dQ, t, axis=0)
    return Q, dQ, ddQ


def foot_points_world(sk, q):
    sk.setPositions(q)
    out = {}
    for side, pts in FOOT_POINTS.items():
        out[side] = np.array([(sk.getBodyNode(b).getWorldTransform().matrix() @ np.r_[o, 1])[:3] for b, o in pts])
    return out


def parse_ranges(text):
    """'12-48,90-95' -> danh sách (12, 48), (90, 95)"""
    out = []
    for part in filter(None, (text or "").split(",")):
        a, _, b = part.partition("-")
        out.append((int(a), int(b or a)))
    return out


def detect_contacts(sk, Q, frame_ids, height, flight):
    heights = {s: [] for s in FOOT_POINTS}
    for q in Q:
        for side, p in foot_points_world(sk, q).items():
            heights[side].append(p[:, 2])
    H = {s: np.array(v) for s, v in heights.items()}  # (N, 4) toạ độ z của từng điểm
    lowest = np.minimum(H["L"].min(axis=1), H["R"].min(axis=1))
    airborne = np.zeros(len(Q), bool)
    for a, b in flight:
        airborne |= (frame_ids >= a) & (frame_ids <= b)
    contact = {}
    for s in FOOT_POINTS:
        foot_low = H[s].min(axis=1)
        on = (foot_low - lowest < height) & ~airborne
        contact[s] = (H[s] - foot_low[:, None] < height) & on[:, None]  # (N, 4) bool
    return contact, airborne, lowest


def solve_frame(sk, q, dq, ddq, contact_pts, mu, w_reg):
    sk.setPositions(q)
    sk.setVelocities(dq)
    M = sk.getMassMatrix()
    h = sk.getCoriolisAndGravityForces()
    n = len(q)
    b = M @ ddq + h
    nc = len(contact_pts)
    r = cp.Variable(N_ROOT)
    tau = cp.Variable(n - N_ROOT)
    cons = []
    rhs = cp.hstack([r, tau])
    cost = cp.sum_squares(r) + w_reg * cp.sum_squares(tau)
    f = None
    if nc:
        Jc = np.vstack([sk.getLinearJacobian(sk.getBodyNode(bn), np.array(o, float).reshape(3, 1))
                        for bn, o in contact_pts])
        f = cp.Variable(3 * nc)
        rhs = rhs + Jc.T @ f
        cost = cost + w_reg * cp.sum_squares(f)
        for k in range(nc):
            fx, fy, fz = f[3 * k], f[3 * k + 1], f[3 * k + 2]
            cons += [fz >= 0, cp.abs(fx) <= mu * fz, cp.abs(fy) <= mu * fz]
    prob = cp.Problem(cp.Minimize(cost), cons + [rhs == b])
    prob.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    fv = np.zeros(0) if f is None else np.asarray(f.value)
    return np.asarray(r.value), np.asarray(tau.value), fv


def cell(text, ratio, sign, flag, tip):
    t = min(abs(ratio), 1.0)
    cls = "c " + ("pos" if sign >= 0 else "neg")
    if t > 0.6:
        cls += " hi"
    if flag:
        cls += " flag"
    return "<td class='{}' style='--t:{:.0f}%' data-v='{:.3f}' title='{}'>{}</td>".format(
        cls, 100 * t, abs(ratio), html.escape(tip), text)


def table(row_labels, headers, rows):
    head = "<tr><th class='r'>frame</th>" + "".join("<th>{}</th>".format(h) for h in headers) + "</tr>"
    body = "".join("<tr><td class='r'>{}</td>{}</tr>".format(l, "".join(r)) for l, r in zip(row_labels, rows))
    return "<table><thead>{}</thead><tbody>{}</tbody></table>".format(head, body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ik_dir", help="thư mục output của run_cometh.py --mode ik (có q.csv)")
    parser.add_argument("--keypoints", required=True, help="CSV keypoint đã đưa vào run_cometh.py")
    parser.add_argument("--q", default="q.csv", help="file góc khớp trong ik_dir")
    parser.add_argument("--out", default=None, help="mặc định: <ik_dir>/dynamics")
    parser.add_argument("--cutoff", type=float, default=6.0, help="[Hz] lọc thông thấp q trước khi lấy đạo hàm")
    parser.add_argument("--flight", default="", help="các frame id đang bay, ví dụ '12-48' hoặc '12-48,120-130'")
    parser.add_argument("--contact-height", type=float, default=0.05,
                        help="[m] bàn chân / điểm bàn chân gần điểm thấp nhất trong khoảng này là đang chạm đất")
    parser.add_argument("--mu", type=float, default=0.8, help="hệ số ma sát tại các điểm tiếp xúc")
    parser.add_argument("--max-grf", type=float, default=6.0, help="[lần trọng lượng] lực đất theo phương đứng")
    parser.add_argument("--max-residual-f", type=float, default=0.10, help="[tỉ lệ so với trọng lượng]")
    parser.add_argument("--max-residual-m", type=float, default=0.05, help="[tỉ lệ so với trọng lượng x 1 m]")
    parser.add_argument("--w-reg", type=float, default=1e-4, help="trọng số nhỏ cho |tau|^2 và |f|^2")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.ik_dir, "dynamics")
    os.makedirs(out_dir, exist_ok=True)
    df_q = pd.read_csv(os.path.join(args.ik_dir, args.q))
    times = df_q.pop("time").values
    Q0 = df_q.values
    n = len(Q0)
    markers = pd.read_csv(args.keypoints).iloc[:n].reset_index(drop=True)
    frame_ids = markers["frame_id"].values if "frame_id" in markers else np.arange(n)

    s, _ = build_skeleton(load_targets(markers)[0])
    sk = s._nimble
    sk.setGravity(np.array([0.0, 0.0, -G]))
    names = dof_names(s)
    mass = sk.getMass()
    bw = mass * G
    fs = 1.0 / np.median(np.diff(times))
    print("Body mass {:.1f} kg, {} frames at {:.0f} Hz".format(mass, n, fs))

    Q, dQ, ddQ = differentiate(Q0, times, args.cutoff)
    flight = parse_ranges(args.flight)
    contact, airborne, lowest_foot = detect_contacts(sk, Q, frame_ids, args.contact_height, flight)
    if not flight:
        print("Warning: no --flight given, a foot is assumed on the ground in every frame")

    TAU = np.full((n, len(names) - N_ROOT), np.nan)
    RES = np.full((n, N_ROOT), np.nan)
    GRF = np.zeros((n, 2, 3))  # theo từng bàn chân, xyz trong hệ world
    for i in range(n):
        pts, owner = [], []
        for k, side in enumerate(FOOT_POINTS):
            for j, p in enumerate(FOOT_POINTS[side]):
                if contact[side][i, j]:
                    pts.append(p)
                    owner.append(k)
        sol = solve_frame(sk, Q[i], dQ[i], ddQ[i], pts, args.mu, args.w_reg)
        if sol is None:
            print("  frame {}: QP failed".format(i))
            continue
        RES[i], TAU[i], f = sol
        for c, k in enumerate(owner):
            GRF[i, k] += f[3 * c:3 * c + 3]
        if (i + 1) % 50 == 0:
            print("  frame {}/{}".format(i + 1, n))

    jn = names[N_ROOT:]
    lim = np.array([torque_limit(x) for x in jn]) * mass  # Nm
    grf_v = GRF[:, :, 2].sum(axis=1)
    grf_h = np.linalg.norm(GRF[:, :, :2].sum(axis=1), axis=1)
    # lực dư: 3 DOF gốc đầu là các phép xoay khung chậu (mô-men), 3 DOF sau là tịnh tiến (lực)
    res_m = np.linalg.norm(RES[:, :3], axis=1)
    res_f = np.linalg.norm(RES[:, 3:], axis=1)
    n_left, n_right = contact["L"].sum(axis=1), contact["R"].sum(axis=1)

    fl_tau = np.abs(TAU) > lim
    fl_grf = grf_v > args.max_grf * bw
    fl_rf = res_f > args.max_residual_f * bw
    fl_rm = res_m > args.max_residual_m * bw

    # CSV
    out = pd.DataFrame(TAU, columns=jn)
    out.insert(0, "frame_id", frame_ids)
    out.insert(1, "time", times)
    out.to_csv(os.path.join(out_dir, "tau.csv"), index=False, float_format="%.3f")
    gr = pd.DataFrame({"frame_id": frame_ids, "time": times, "contact_points_left": n_left,
                       "contact_points_right": n_right, "airborne": airborne, "lowest_foot_z": lowest_foot,
                       "grf_left_z": GRF[:, 0, 2], "grf_right_z": GRF[:, 1, 2], "grf_vertical_bw": grf_v / bw,
                       "grf_horizontal_bw": grf_h / bw, "residual_force_N": res_f, "residual_moment_Nm": res_m})
    gr.to_csv(os.path.join(out_dir, "grf_residual.csv"), index=False, float_format="%.4f")
    rows = []
    for i, j in zip(*np.nonzero(fl_tau)):
        rows.append({"frame_id": frame_ids[i], "time": times[i], "issue": "torque", "item": jn[j],
                     "value": TAU[i, j], "unit": "Nm", "per_kg": TAU[i, j] / mass, "limit": lim[j]})
    for mask, issue, val, unit, limit in [(fl_grf, "grf", grf_v / bw, "BW", args.max_grf),
                                          (fl_rf, "residual_f", res_f, "N", args.max_residual_f * bw),
                                          (fl_rm, "residual_m", res_m, "Nm", args.max_residual_m * bw)]:
        for i in np.nonzero(mask)[0]:
            rows.append({"frame_id": frame_ids[i], "time": times[i], "issue": issue, "item": "",
                         "value": val[i], "unit": unit, "per_kg": np.nan, "limit": limit})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "flags_dynamics.csv"), index=False)

    # HTML
    labels = ["{} <small>(#{}, {:.2f}s)</small>".format(i, frame_ids[i], times[i]) for i in range(n)]
    cols = [j for j in range(len(jn)) if np.nanmax(np.abs(TAU[:, j])) > 0.5]
    cols.sort(key=lambda j: -np.nanmax(np.abs(TAU[:, j]) / lim[j]))
    tau_rows = []
    for i in range(n):
        tau_rows.append([cell("{:.0f}".format(TAU[i, j]), TAU[i, j] / lim[j], TAU[i, j], fl_tau[i, j],
                              "frame {} · {}: {:.1f} Nm = {:.2f} Nm/kg (ngưỡng ±{:.1f} Nm/kg)".format(
                                  frame_ids[i], jn[j], TAU[i, j], TAU[i, j] / mass, lim[j] / mass))
                         for j in cols])
    tab1 = table(labels, ["{}<br><small>±{:.0f}</small>".format(html.escape(jn[j]), lim[j]) for j in cols], tau_rows)

    gr_rows = []
    for i in range(n):
        side = "".join(x for x, c in (("T", n_left[i]), ("P", n_right[i])) if c) or "bay"
        gr_rows.append([
            "<td>{}</td>".format(side),
            cell("{:.2f}".format(grf_v[i] / bw), grf_v[i] / bw / args.max_grf, 1, fl_grf[i],
                 "frame {} · lực đất dọc {:.0f} N = {:.2f} × trọng lượng (trái {:.0f} N, phải {:.0f} N)".format(
                     frame_ids[i], grf_v[i], grf_v[i] / bw, GRF[i, 0, 2], GRF[i, 1, 2])),
            cell("{:.2f}".format(grf_h[i] / bw), grf_h[i] / bw / args.max_grf, 1, False,
                 "frame {} · lực đất ngang {:.0f} N".format(frame_ids[i], grf_h[i])),
            cell("{:.0f}".format(res_f[i]), res_f[i] / (args.max_residual_f * bw), 1, fl_rf[i],
                 "frame {} · lực dư {:.0f} N = {:.0f}% trọng lượng".format(frame_ids[i], res_f[i],
                                                                          100 * res_f[i] / bw)),
            cell("{:.0f}".format(res_m[i]), res_m[i] / (args.max_residual_m * bw), 1, fl_rm[i],
                 "frame {} · mô-men dư {:.0f} Nm".format(frame_ids[i], res_m[i])),
        ])
    tab2 = table(labels, ["tiếp xúc<br><small>T=trái P=phải</small>",
                          "lực đất dọc<br><small>× trọng lượng, ngưỡng {:g}</small>".format(args.max_grf),
                          "lực đất ngang<br><small>× trọng lượng</small>",
                          "lực dư [N]<br><small>ngưỡng {:.0f}</small>".format(args.max_residual_f * bw),
                          "mô-men dư [Nm]<br><small>ngưỡng {:.0f}</small>".format(args.max_residual_m * bw)],
                 gr_rows)

    ratio = np.abs(TAU) / lim
    order = np.dstack(np.unravel_index(np.argsort(-np.nan_to_num(ratio), axis=None)[:12], ratio.shape))[0]
    top = "".join("<tr><td>#{}</td><td>{}</td><td>{:+.0f} Nm</td><td>{:+.2f}</td><td>±{:.1f}</td><td>{:.1f}×</td></tr>"
                  .format(frame_ids[i], jn[j], TAU[i, j], TAU[i, j] / mass, lim[j] / mass, ratio[i, j])
                  for i, j in order)
    n_flight = int(((n_left == 0) & (n_right == 0)).sum())
    tiles = [("{:.1f} kg".format(mass), "khối lượng mô hình"),
             ("{}".format(int(fl_tau.sum())), "ô mô-men vượt ngưỡng"),
             ("{}".format(int((fl_rf | fl_rm).sum())), "frame có lực / mô-men dư lớn"),
             ("{:.1f}×".format(np.nanmax(grf_v) / bw), "lực đất dọc lớn nhất (× trọng lượng)"),
             ("{}".format(n_flight), "frame không chạm đất")]
    sections = []
    for title, tbl, note, th in [
        ("Mô-men khớp", tab1, "Ô = mô-men tại khớp (Nm), dấu theo chiều DOF. Màu đậm dần theo tỉ lệ với ngưỡng "
         "của khớp đó; viền vàng = vượt ngưỡng. Cột xếp theo tỉ lệ lớn nhất.", 0.0),
        ("Lực đất & lực dư", tab2, "Lực dư = lực / mô-men phải \"bịa\" thêm ở khung chậu để chuyển động khớp "
         "với vật lý. Lớn nghĩa là chuyển động (hoặc chỗ tiếp đất) không hợp lý về mặt vật lý.", 0.0)]:
        sections.append("""<section class="tab" hidden>
<div class="controls"><label>Ẩn ô có tỉ lệ ngưỡng nhỏ hơn <input class="th" type="number" step="0.1" min="0"
 value="{th:g}" oninput="apply()"></label><label><input class="only" type="checkbox" onchange="apply()"> chỉ hiện
 frame có ô vượt</label><span><span class='swatch'></span> vượt ngưỡng</span></div><p class="note">{note}</p>
<div class="wrap">{tbl}</div></section>""".format(th=th, note=html.escape(note), tbl=tbl))

    doc = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Inverse dynamics</title>
<style>{css}</style><script>{js}</script></head><body>
<h1>Inverse dynamics: mô-men và lực</h1>
<p class="sub">{n} frame · góc khớp từ {qfile} (không sửa), đạo hàm sau lọc {cut:g} Hz · tiếp đất khi điểm gót / mũi
chân cách điểm thấp nhất &lt; {ch:g} cm, pha bay khai báo: {fl} · μ = {mu:g}. Ngưỡng mô-men là mức tham khảo thô cho vận động thể thao.</p>
<div class="tiles">{tiles}</div>
<h2 style="font-size:16px">12 mô-men vượt ngưỡng nhiều nhất</h2>
<div class="wrap" style="max-height:none;display:inline-block"><table class="top"><thead><tr><th>frame id</th>
<th>DOF</th><th>mô-men</th><th>Nm/kg</th><th>ngưỡng Nm/kg</th><th>tỉ lệ</th></tr></thead><tbody>{top}</tbody></table></div>
<h2 style="font-size:16px">Chi tiết từng frame</h2>
<div class="tabs"><button onclick="show(0)">Mô-men khớp</button><button onclick="show(1)">Lực đất &amp; lực dư</button></div>
{sections}</body></html>""".format(
        css=CSS, js=JS % json.dumps(["Mô-men khớp", "Lực đất & lực dư"]), n=n, qfile=html.escape(args.q),
        cut=args.cutoff, ch=100 * args.contact_height, mu=args.mu, fl=html.escape(args.flight or "không có"), top=top,
        tiles="".join("<div class='tile'><b>{}</b><span>{}</span></div>".format(a, html.escape(b)) for a, b in tiles),
        sections="".join(sections))
    with open(os.path.join(out_dir, "dynamics.html"), "w", encoding="utf-8") as f:
        f.write(doc)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(times, GRF[:, 0, 2] / bw, label="left")
        axes[0].plot(times, GRF[:, 1, 2] / bw, label="right")
        axes[0].set_ylabel("vertical GRF / BW")
        axes[0].legend(fontsize=8)
        axes[1].plot(times, res_f / bw * 100)
        axes[1].axhline(args.max_residual_f * 100, color="gray", ls="--", lw=0.8)
        axes[1].set_ylabel("residual force [% BW]")
        for p in ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l"]:
            if p in jn:
                axes[2].plot(times, TAU[:, jn.index(p)] / mass, label=p)
        axes[2].set_ylabel("torque [Nm/kg]")
        axes[2].set_xlabel("time [s]")
        axes[2].legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dynamics.png"), dpi=110)
        plt.close(fig)
    except ImportError:
        pass

    print("Torques above the limit: {} cells in {} frames".format(int(fl_tau.sum()), int(fl_tau.any(axis=1).sum())))
    print("Frames: GRF > {:g} BW: {}, residual force > {:.0f} N: {}, residual moment > {:.0f} Nm: {}, "
          "no contact: {}".format(args.max_grf, int(fl_grf.sum()), args.max_residual_f * bw, int(fl_rf.sum()),
                                  args.max_residual_m * bw, int(fl_rm.sum()), n_flight))
    print("Results saved in", out_dir)


if __name__ == "__main__":
    main()
