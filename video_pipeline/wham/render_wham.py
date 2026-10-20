"""
Render một wham_output.pkl giống demo của WHAM (video input có phủ mesh | góc nhìn toàn cục có mặt đất), từ các
đỉnh đã lưu, với faces và J_regressor của model SMPL mà WHAM dùng (SMPL_NEUTRAL.pkl) và repo WHAM để dùng
renderer pytorch3d của nó.

    python video_pipeline/wham/render_wham.py video_pipeline/wham/out/x/ik_exact/refined/wham_output_refined.pkl \
        --video wham_input.mp4 --compare-pkl path/to/wham_output.pkl \
        --flags video_pipeline/wham/out/x/ik_exact/refined/flags.csv

Output (cạnh file pkl nếu không có -o; chọn bằng --outputs):
    wham_refined.mp4   cùng bố cục với video output của WHAM, với mesh được truyền vào
    wham_compare.mp4   (cần --compare-pkl) trên: pkl còn lại (WHAM gốc), dưới: pkl được truyền vào, nửa kích thước,
                       hai hàng dùng chung mặt đất và camera toàn cục; có --flags thì liệt kê các DOF bị refine_q.py
                       gắn cờ trong frame (trên: góc gốc, dưới: sau khi sửa)
    wham_overlay.mp4   (cần --compare-pkl) hai mesh trong cùng một cảnh, pkl còn lại màu xám, pkl được truyền vào
                       màu --overlay-color (hoặc tô theo độ lệch với --heat): chỗ nào thấy mesh màu thì mesh đó
                       nằm trước mesh gốc
"""
import argparse
import os
import sys

import cv2
import imageio
import joblib
import numpy as np
import torch

from overlay_video import draw_text_block, flag_lines
from wham_to_keypoints import DEFAULT_SMPL, camera_to_world, load_smpl, to_world


def load_subject(path, subject):
    results = joblib.load(path)
    sid = subject if subject is not None else max(results, key=lambda k: len(results[k]["frame_ids"]))
    return results[sid]


def global_verts(d, J):
    """Các đỉnh trong hệ world của WHAM (= SMPL(pose_world, trans_world)), như trong run_vis của WHAM."""
    _, R_wc, pelvis_cam, pelvis_world, _ = camera_to_world(d, J, "none")
    return to_world(d["verts"].astype(np.float64), pelvis_cam, R_wc, pelvis_world)


def rgb(c):
    return tuple(reversed(c))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl")
    parser.add_argument("--video", required=True, help="video đã đưa vào WHAM")
    parser.add_argument("--smpl-model", default=DEFAULT_SMPL, help="SMPL_NEUTRAL.pkl, giống lúc chạy wham_to_keypoints.py")
    parser.add_argument("--compare-pkl", default=None, help="pkl thứ hai (WHAM gốc) cho wham_compare.mp4")
    parser.add_argument("--flags", default=None, help="flags.csv của refine_q.py, liệt kê trong wham_compare.mp4")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--wham-root", default=os.path.expanduser("~/projects/WHAM"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-lines", type=int, default=8)
    parser.add_argument("--outputs", default="refined,compare,overlay",
                        help="một hoặc nhiều trong refined, compare, overlay, cách nhau bởi dấu phẩy")
    parser.add_argument("--overlay-color", default="0.95,0.45,0.15", help="màu RGB trong [0, 1] của mesh được truyền vào")
    parser.add_argument("--heat", type=float, default=0.0,
                        help="[mm] tô mesh được truyền vào theo độ lệch, đậm nhất ở giá trị này (0: tắt)")
    parser.add_argument("-o", "--out-dir", default=None)
    args = parser.parse_args()

    sys.path.insert(0, args.wham_root)
    from lib.vis.renderer import Renderer, get_global_cameras, overlay_image_onto_background
    from pytorch3d.renderer import Materials, TexturesVertex
    from pytorch3d.structures import Meshes, join_meshes_as_scene

    dev = torch.device(args.device)
    J, _, faces = load_smpl(args.smpl_model)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.pkl))

    runs = [("SAU KHI SUA", load_subject(args.pkl, args.subject), "after")]
    if args.compare_pkl:
        runs.insert(0, ("WHAM GOC", load_subject(args.compare_pkl, args.subject), "before"))

    flags = {"before": {}, "after": {}}
    if args.flags:
        import pandas as pd
        for f in pd.read_csv(args.flags).to_dict("records"):
            flags[f["stage"]].setdefault(int(f["frame_id"]), []).append(f)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal = (width ** 2 + height ** 2) ** 0.5  # như trong run_vis của WHAM

    # Mặt đất và camera toàn cục lấy từ pkl đầu tiên (pkl gốc nếu có --compare-pkl), dùng chung cho mọi hàng
    vg = [torch.from_numpy(global_verts(d, J)).float() for _, d, _ in runs]
    y0 = vg[0][..., 1].min()
    for v in vg:
        v[..., 1] -= y0
    ref = vg[0]
    cx, cz = (ref.mean(1).max(0)[0] + ref.mean(1).min(0)[0])[[0, 2]] / 2.0
    sx, sz = (ref.mean(1).max(0)[0] - ref.mean(1).min(0)[0])[[0, 2]]
    renderer = Renderer(width, height, focal, dev, faces)
    create_renderer = renderer.create_renderer

    def create_renderer_big_bins():
        # hai mesh làm tràn bin thô của pytorch3d (ảnh bị thủng lỗ); rasterizer không chia bin thì quá chậm
        create_renderer()
        renderer.renderer.rasterizer.raster_settings.max_faces_per_bin = 100000

    renderer.create_renderer = create_renderer_big_bins
    renderer.create_renderer()
    renderer.set_ground(max(sx.item(), sz.item()) * 1.5, cx.item(), cz.item())
    global_R, global_T, global_lights = get_global_cameras(ref, dev)
    row_of = [{int(f): r for r, f in enumerate(d["frame_ids"])} for _, d, _ in runs]

    def render(k, img, frame_i):
        r = row_of[k].get(frame_i)
        if r is None:
            return np.concatenate((img, np.ones_like(img) * 255), axis=1), None
        d = runs[k][1]
        renderer.create_camera(torch.eye(3), torch.zeros(3))
        left = renderer.render_mesh(torch.from_numpy(d["verts"][r]).float().to(dev), img)
        colors = torch.ones((1, 4)).float().to(dev)
        colors[..., :3] *= 0.9
        cameras = renderer.create_camera(global_R[r], global_T[r])
        right = renderer.render_with_ground(vg[k][[r]].to(dev), renderer.faces.clone().squeeze(0), colors, cameras,
                                            global_lights)
        return np.concatenate((left, right), axis=1), r

    gray = [0.8, 0.8, 0.8]
    color = [float(c) for c in args.overlay_color.split(",")]

    def vertex_colors(r0, r1):
        """(2, V, 3) màu của mesh gốc và mesh được truyền vào."""
        V = runs[0][1]["verts"].shape[1]
        c0 = torch.tensor(gray).expand(V, 3)
        if args.heat > 0:
            disp = np.linalg.norm(runs[1][1]["verts"][r1] - runs[0][1]["verts"][r0], axis=1) * 1000
            t = torch.from_numpy(np.clip(disp / args.heat, 0, 1)).float()[:, None]
            c1 = (1 - t) * torch.tensor(gray) + t * torch.tensor(color)
        else:
            c1 = torch.tensor(color).expand(V, 3)
        return torch.stack([c0, c1]).float().to(dev)

    def render_both(img, frame_i):
        """Hai mesh trong cùng một cảnh (dùng chung độ sâu): góc nhìn camera | góc nhìn toàn cục."""
        r0, r1 = row_of[0].get(frame_i), row_of[1].get(frame_i)
        if r0 is None or r1 is None:
            return np.concatenate((img, np.ones_like(img) * 255), axis=1), None, None
        verts = [torch.from_numpy(runs[0][1]["verts"][r0]).float().to(dev),
                 torch.from_numpy(runs[1][1]["verts"][r1]).float().to(dev)]
        cols = vertex_colors(r0, r1)
        f = renderer.faces.squeeze(0)
        renderer.create_camera(torch.eye(3), torch.zeros(3))
        renderer.update_bbox(torch.cat(verts)[::50], scale=1.2)
        scene = join_meshes_as_scene(Meshes(verts=verts, faces=[f, f], textures=TexturesVertex(list(cols))))
        res = torch.flip(renderer.renderer(scene, materials=Materials(device=dev, specular_color=(gray,), shininess=0),
                                           cameras=renderer.cameras, lights=renderer.lights), [1, 2])
        left = overlay_image_onto_background(res[0, ..., :3] * 255, res[0, ..., -1] > 1e-3, renderer.bboxes,
                                             img.copy())
        renderer.reset_bbox()
        # góc nhìn toàn cục có mặt đất, như render_with_ground của WHAM nhưng với hai mesh
        cameras = renderer.create_camera(global_R[r0], global_T[r0])
        gv, gf, gc = renderer.ground_geometry
        scene = join_meshes_as_scene(Meshes(
            verts=[vg[0][r0].to(dev), vg[1][r1].to(dev), gv], faces=[f, f, gf],
            textures=TexturesVertex([cols[0] * 0.9 / 0.8, cols[1], gc[..., :3].float()])))
        res = renderer.renderer(scene, cameras=cameras, lights=global_lights,
                                materials=Materials(device=dev, shininess=0))
        right = (res[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        return np.concatenate((left, right), axis=1), r0, r1

    outputs = set(args.outputs.split(","))
    if not args.compare_pkl:
        outputs &= {"refined"}

    def writer(name):
        return imageio.get_writer(os.path.join(out_dir, name), fps=fps, mode="I", format="FFMPEG",
                                  macro_block_size=1)

    w_main = writer("wham_refined.mp4") if "refined" in outputs else None
    w_cmp = writer("wham_compare.mp4") if "compare" in outputs else None
    w_ovl = writer("wham_overlay.mp4") if "overlay" in outputs else None
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = frame[..., ::-1].copy()
        rows = []
        for k, (title, _, stage) in enumerate(runs):
            if w_cmp is None and (w_main is None or k < len(runs) - 1):
                continue
            out, r = render(k, img, frame_i)
            out = np.ascontiguousarray(out.astype(np.uint8))
            if k == len(runs) - 1 and w_main is not None:
                w_main.append_data(out)
            if w_cmp is not None:
                small = np.ascontiguousarray(cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2),
                                                        interpolation=cv2.INTER_AREA))
                lines = [(title, (255, 255, 255)), ("frame {}".format(frame_i), (220, 220, 220))]
                if r is not None and args.flags:
                    key = "value_deg" if stage == "before" else "refined_deg"
                    lines += [(t, rgb(c)) for t, c in flag_lines(flags[stage].get(frame_i, []), key, args.max_lines)]
                draw_text_block(small, lines, scale=0.45)
                rows.append(small)
        if w_cmp is not None:
            w_cmp.append_data(np.vstack(rows))
        if w_ovl is not None:
            out, r0, r1 = render_both(img, frame_i)
            out = np.ascontiguousarray(out.astype(np.uint8))
            label = "da sua (dam = lech >= {:g} mm)".format(args.heat) if args.heat > 0 else "cam: da sua"
            lines = [("xam: WHAM goc", (230, 230, 230)), (label, tuple(int(255 * c) for c in color)),
                     ("frame {}".format(frame_i), (220, 220, 220))]
            if r0 is not None:
                d_mm = 1000 * np.linalg.norm(runs[1][1]["verts"][r1] - runs[0][1]["verts"][r0], axis=1)
                lines.append(("lech trung binh {:.0f} mm, max {:.0f} mm".format(d_mm.mean(), d_mm.max()),
                              (220, 220, 220)))
                if args.flags:
                    lines += [(t, rgb(c)) for t, c in flag_lines(flags["before"].get(frame_i, []), "value_deg",
                                                                 args.max_lines)]
            draw_text_block(out, lines, scale=0.9)
            w_ovl.append_data(out)
        frame_i += 1
        if frame_i % 50 == 0:
            print("  frame", frame_i)
    for w in (w_main, w_cmp, w_ovl):
        if w is not None:
            w.close()
    print("Saved in {}: {}".format(out_dir, ", ".join("wham_{}.mp4".format(o) for o in sorted(outputs))))


if __name__ == "__main__":
    main()
