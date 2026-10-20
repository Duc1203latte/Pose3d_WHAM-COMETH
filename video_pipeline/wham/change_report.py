"""
Bảng những gì refine_q.py + wham_refine_mesh.py đã thay đổi, theo từng frame: một trang HTML có ba tab,
kèm cùng số liệu dưới dạng CSV.

    python video_pipeline/wham/change_report.py --refined video_pipeline/wham/out/x/ik_exact/refined \
        --wham-pkl path/to/wham_output.pkl [--smpl-model path/to/SMPL_NEUTRAL.pkl]

Các tab của changes.html (ô tô màu theo mức thay đổi, rê chuột lên ô để xem trước / sau):
    Góc khớp BSM     sau - trước của mọi góc khớp COMETH [độ]; ô bị refine_q.py gắn cờ có viền
    Khớp SMPL        góc giữa phép xoay cục bộ gốc và đã sửa của từng khớp SMPL [độ]
    Mesh             dịch chuyển trung bình của các đỉnh thuộc từng bộ phận cơ thể SMPL [mm]
Output trong --refined: changes.html, changes_bsm.csv, changes_smpl.csv, changes_mesh.csv
"""
import argparse
import html
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from wham_to_keypoints import DEFAULT_SMPL, load_smpl

N_ROOT = 6
SMPL_JOINTS = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
               "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_hand", "right_hand"]
ISSUE_VI = {"out_of_rom": "ngoài ROM", "at_limit": "dính biên", "spike": "đột biến", "too_fast": "quá nhanh"}


def load_subject(path):
    results = joblib.load(path)
    return results[max(results, key=lambda k: len(results[k]["frame_ids"]))]


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--head:#f4f3f0;
--mid:#f0efec;--neg:#2a78d6;--pos:#e34948;--seq:#104281;--flag:#eda100;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--line:#383835;--head:#242423;--mid:#383835;--neg:#3987e5;--pos:#e66767;--seq:#86b6ef;--flag:#c98500;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--head:#242423;--mid:#383835;
--neg:#3987e5;--pos:#e66767;--seq:#86b6ef;--flag:#c98500;color-scheme:dark}
body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--ink)}
h1{font-size:20px;margin:0 0 4px}p.sub{color:var(--ink2);margin:0 0 16px;font-size:14px}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
.tile{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:150px}
.tile b{display:block;font-size:22px}.tile span{color:var(--ink2);font-size:13px}
.tabs{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:10px}
.tabs button{border:0;background:none;color:var(--ink2);padding:8px 12px;font:inherit;cursor:pointer;
border-bottom:2px solid transparent}.tabs button.on{color:var(--ink);border-color:var(--ink)}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:14px;margin:8px 0;color:var(--ink2)}
.controls input[type=number]{width:64px}
.note{font-size:13px;color:var(--ink2);margin:4px 0 8px}
.wrap{max-height:70vh;overflow:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:separate;border-spacing:0;font-size:12px}
th,td{padding:3px 6px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--head);z-index:2;font-weight:600;vertical-align:bottom}
th.r,td.r{position:sticky;left:0;background:var(--head);text-align:left;z-index:1}th.r{z-index:3}
td.c{background:color-mix(in oklab,var(--hue) var(--t),var(--mid));font-variant-numeric:tabular-nums}
td.c.neg{--hue:var(--neg)}td.c.pos{--hue:var(--pos)}td.c.seq{--hue:var(--seq)}
td.c.hi{color:#fff}td.c.lo{color:transparent;background:transparent}
td.c.flag{box-shadow:inset 0 0 0 2px var(--flag)}
tr.hide{display:none}
.legend{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink2)}
.bar{width:160px;height:10px;border-radius:2px}
.swatch{display:inline-block;width:14px;height:14px;border-radius:2px;box-shadow:inset 0 0 0 2px var(--flag);vertical-align:middle}
table.top td{text-align:left}
"""

JS = """
const TABS = %s;
let cur = 0;
function show(i){cur=i;document.querySelectorAll('.tabs button').forEach((b,j)=>b.classList.toggle('on',j===i));
  document.querySelectorAll('section.tab').forEach((s,j)=>s.hidden=j!==i);apply();}
function apply(){const s=document.querySelectorAll('section.tab')[cur];
  const th=parseFloat(s.querySelector('.th').value)||0, only=s.querySelector('.only').checked;
  s.querySelectorAll('tbody tr').forEach(tr=>{let any=false;
    tr.querySelectorAll('td.c').forEach(td=>{const small=Math.abs(+td.dataset.v)<th;td.classList.toggle('lo',small);
      if(!small)any=true;});
    tr.classList.toggle('hide',only&&!any);});}
document.addEventListener('DOMContentLoaded',()=>show(0));
"""


def heat_table(row_labels, col_labels, values, tips, diverging, scale, flags=None, unit=""):
    """values (N, C); tips (N, C) chuỗi tooltip; scale = giá trị ứng với màu đậm nhất."""
    head = "<tr><th class='r'>frame</th>" + "".join("<th>{}</th>".format(html.escape(c)) for c in col_labels) + "</tr>"
    body = []
    for i, label in enumerate(row_labels):
        cells = []
        for j in range(len(col_labels)):
            v = values[i, j]
            t = min(abs(v) / scale, 1.0) if scale > 0 else 0
            cls = ("pos" if v > 0 else "neg") if diverging else "seq"
            if t > 0.6:
                cls += " hi"
            if flags is not None and flags[i][j]:
                cls += " flag"
            cells.append("<td class='c {}' style='--t:{:.0f}%' data-v='{:.3f}' title='{}'>{}</td>".format(
                cls, 100 * t, v, html.escape(tips[i][j]), "{:+.1f}".format(v) if diverging else "{:.1f}".format(v)))
        body.append("<tr><td class='r'>{}</td>{}</tr>".format(label, "".join(cells)))
    return "<table><thead>{}</thead><tbody>{}</tbody></table>".format(head, "".join(body))


def legend(diverging, scale, unit):
    if diverging:
        grad = "linear-gradient(90deg,var(--neg),var(--mid),var(--pos))"
        labels = "<span>−{0:g}{1}</span><div class='bar' style='background:{2}'></div><span>+{0:g}{1}</span>".format(
            scale, unit, grad)
    else:
        grad = "linear-gradient(90deg,var(--mid),var(--seq))"
        labels = "<span>0</span><div class='bar' style='background:{1}'></div><span>≥ {0:g}{2}</span>".format(
            scale, grad, unit)
    return "<div class='legend'>{}</div>".format(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refined", required=True, help="thư mục output của refine_q.py")
    parser.add_argument("--ik-dir", default=None, help="thư mục chứa q.csv gốc (mặc định: thư mục cha của --refined)")
    parser.add_argument("--wham-pkl", required=True, help="wham_output.pkl gốc")
    parser.add_argument("--refined-pkl", default=None, help="mặc định: <refined>/wham_output_refined.pkl")
    parser.add_argument("--smpl-model", default=DEFAULT_SMPL,
                        help="SMPL_NEUTRAL.pkl (skin weights -> bộ phận cơ thể)")
    args = parser.parse_args()

    ik_dir = args.ik_dir or os.path.dirname(os.path.abspath(args.refined.rstrip("/\\")))
    q0 = pd.read_csv(os.path.join(ik_dir, "q.csv"))
    q1 = pd.read_csv(os.path.join(args.refined, "q_refined.csv"))
    times = q0.pop("time").values
    q1.pop("time")
    names = list(q0.columns)
    A0, A1 = np.degrees(q0.values), np.degrees(q1.values)
    flags = pd.read_csv(os.path.join(args.refined, "flags.csv"))
    flags = flags[flags.stage == "before"]

    d0 = load_subject(args.wham_pkl)
    d1 = load_subject(args.refined_pkl or os.path.join(args.refined, "wham_output_refined.pkl"))
    frame_ids = np.asarray(d0["frame_ids"])
    n = min(len(A0), len(frame_ids))
    _, W, _ = load_smpl(args.smpl_model)
    part = W.argmax(axis=1)

    # --- Góc khớp BSM
    dA = (A1 - A0)[:n]
    cols = [j for j in range(N_ROOT, len(names)) if np.abs(dA[:, j]).max() >= 0.05]
    cols.sort(key=lambda j: -np.abs(dA[:, j]).max())
    flag_set = {}
    for f in flags.itertuples():
        flag_set.setdefault((int(f.frame), f.dof), []).append(ISSUE_VI.get(f.issue, f.issue))
    bsm_tips, bsm_flags = [], []
    for i in range(n):
        row_t, row_f = [], []
        for j in cols:
            fl = flag_set.get((i, names[j]), [])
            row_t.append("frame {} · {}: trước {:.1f}° → sau {:.1f}° (Δ {:+.1f}°){}".format(
                frame_ids[i], names[j], A0[i, j], A1[i, j], dA[i, j], " · cờ: " + ", ".join(fl) if fl else ""))
            row_f.append(bool(fl))
        bsm_tips.append(row_t)
        bsm_flags.append(row_f)

    # --- Phép xoay cục bộ của các khớp SMPL
    R0 = Rotation.from_rotvec(d0["pose"][:n].reshape(-1, 3))
    R1 = Rotation.from_rotvec(d1["pose"][:n].reshape(-1, 3))
    dRot = np.degrees((R1 * R0.inv()).magnitude()).reshape(n, 24)[:, 1:]
    smpl_names = SMPL_JOINTS[1:]

    # --- Dịch chuyển mesh theo bộ phận cơ thể
    disp = np.linalg.norm(d1["verts"][:n] - d0["verts"][:n], axis=2) * 1000  # (n, V) mm
    parts = [k for k in range(24) if (part == k).any()]
    dMesh = np.stack([disp[:, part == k].mean(axis=1) for k in parts], axis=1)
    mesh_names = [SMPL_JOINTS[k] for k in parts]
    whole = disp.mean(axis=1)

    # CSV
    rows = []
    for i in range(n):
        for j in cols:
            rows.append({"frame": i, "frame_id": frame_ids[i], "time": times[i], "dof": names[j],
                         "before_deg": A0[i, j], "after_deg": A1[i, j], "delta_deg": dA[i, j],
                         "flags": ";".join(flag_set.get((i, names[j]), []))})
    pd.DataFrame(rows).to_csv(os.path.join(args.refined, "changes_bsm.csv"), index=False)
    df = pd.DataFrame(dRot, columns=smpl_names)
    df.insert(0, "frame_id", frame_ids[:n])
    df.to_csv(os.path.join(args.refined, "changes_smpl.csv"), index=False, float_format="%.3f")
    df = pd.DataFrame(dMesh, columns=mesh_names)
    df.insert(0, "frame_id", frame_ids[:n])
    df.insert(1, "whole_body_mm", whole)
    df.to_csv(os.path.join(args.refined, "changes_mesh.csv"), index=False, float_format="%.2f")

    labels = ["{} <small>(#{}, {:.2f}s)</small>".format(i, frame_ids[i], times[i]) for i in range(n)]

    def scale_of(x, floor):
        return max(float(np.percentile(np.abs(x), 99)), floor)

    s_bsm, s_smpl, s_mesh = scale_of(dA[:, cols], 1), scale_of(dRot, 1), scale_of(dMesh, 5)
    tabs = [
        ("Góc khớp BSM", "°", 1.0, True, s_bsm,
         heat_table(labels, [names[j] for j in cols], dA[:, cols], bsm_tips, True, s_bsm, bsm_flags),
         "Ô = góc sau khi sửa − trước khi sửa (độ). Đỏ: tăng, xanh: giảm. Viền vàng: ô bị refine_q.py gắn cờ. "
         "Cột xếp theo mức thay đổi lớn nhất. Chỉ hiện các DOF có thay đổi."),
        ("Khớp SMPL", "°", 1.0, False, s_smpl,
         heat_table(labels, smpl_names, dRot, [["frame {} · {}: xoay thêm {:.1f}°".format(frame_ids[i], c, dRot[i, j])
                                                for j, c in enumerate(smpl_names)] for i in range(n)],
                    False, s_smpl),
         "Ô = góc giữa phép xoay cục bộ cũ và mới của từng khớp SMPL (độ, luôn ≥ 0). Hướng gốc (pelvis) giữ nguyên."),
        ("Mesh", " mm", 5.0, False, s_mesh,
         heat_table(labels, ["cả người"] + mesh_names, np.column_stack([whole, dMesh]),
                    [["frame {} · {}: dịch trung bình {:.1f} mm".format(frame_ids[i], c, v)
                      for c, v in zip(["cả người"] + mesh_names, np.r_[whole[i], dMesh[i]])] for i in range(n)],
                    False, s_mesh),
         "Ô = dịch chuyển trung bình của các đỉnh mesh thuộc từng bộ phận (mm), so với mesh WHAM gốc."),
    ]

    top = np.dstack(np.unravel_index(np.argsort(-np.abs(dA[:, cols]), axis=None)[:12], (n, len(cols))))[0]
    top_rows = "".join("<tr><td>#{}</td><td>{}</td><td>{:.1f}°</td><td>{:.1f}°</td><td>{:+.1f}°</td><td>{}</td></tr>".format(
        frame_ids[i], names[cols[j]], A0[i, cols[j]], A1[i, cols[j]], dA[i, cols[j]],
        ", ".join(flag_set.get((i, names[cols[j]]), [])) or "—") for i, j in top)
    worst = int(np.argmax(whole))
    tiles = [
        ("{}".format(len(flags)), "ô bị gắn cờ trước khi sửa"),
        ("{:.1f}°".format(np.abs(dA[:, cols]).mean()), "thay đổi góc BSM trung bình"),
        ("{:.1f}°".format(np.abs(dA[:, cols]).max()), "thay đổi góc BSM lớn nhất"),
        ("{:.1f} mm".format(whole.mean()), "mesh dịch trung bình"),
        ("{:.0f} mm".format(whole[worst]), "frame #{} dịch nhiều nhất".format(frame_ids[worst])),
    ]

    sections = []
    for title, unit, th, div, sc, table, note in tabs:
        sections.append("""<section class="tab" hidden>
<div class="controls"><label>Ẩn ô nhỏ hơn <input class="th" type="number" step="0.5" min="0" value="{th:g}"
 oninput="apply()">{unit}</label><label><input class="only" type="checkbox" onchange="apply()"> chỉ hiện frame có
 ô vượt ngưỡng</label>{legend}{flag}</div><p class="note">{note}</p><div class="wrap">{table}</div></section>""".format(
            th=th, unit=unit.strip(), legend=legend(div, round(sc, 1), unit.strip()),
            flag="<span><span class='swatch'></span> đã gắn cờ</span>" if div else "", note=html.escape(note),
            table=table))

    doc = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Bảng thay đổi</title>
<style>{css}</style><script>{js}</script></head><body>
<h1>Những gì đã sửa</h1>
<p class="sub">{n} frame · so sánh WHAM / COMETH gốc với bản đã sửa. Rê chuột lên ô để xem giá trị trước / sau.</p>
<div class="tiles">{tiles}</div>
<h2 style="font-size:16px">12 thay đổi góc khớp lớn nhất</h2>
<div class="wrap" style="max-height:none;display:inline-block"><table class="top"><thead><tr><th>frame id</th><th>DOF</th>
<th>trước</th><th>sau</th><th>Δ</th><th>cờ</th></tr></thead><tbody>{top}</tbody></table></div>
<h2 style="font-size:16px">Chi tiết từng frame</h2>
<div class="tabs">{buttons}</div>{sections}</body></html>""".format(
        css=CSS, js=JS % json.dumps([t[0] for t in tabs]), n=n,
        tiles="".join("<div class='tile'><b>{}</b><span>{}</span></div>".format(a, html.escape(b)) for a, b in tiles),
        top=top_rows, buttons="".join("<button onclick='show({})'>{}</button>".format(i, html.escape(t[0]))
                                      for i, t in enumerate(tabs)),
        sections="".join(sections))
    out = os.path.join(args.refined, "changes.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print("Saved", out, "(+ changes_bsm.csv, changes_smpl.csv, changes_mesh.csv)")


if __name__ == "__main__":
    main()
