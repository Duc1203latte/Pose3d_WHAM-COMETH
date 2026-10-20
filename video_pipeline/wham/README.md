# Nhánh WHAM: output WHAM → COMETH → kiểm tra, sửa, video, bảng

```
wham_output.pkl ─[wham_to_keypoints.py]─► wham_keypoints.csv ─[../run_cometh.py --ik exact]─► ik_exact/q.csv
                                                                                                │
      ┌─────────────────────────────────────────────────────────────────────────────────────────┤
      │ refine_q.py        kiểm tra + sửa góc khớp → ik_exact/refined/ (q_refined.csv, report.html, ...)
      │ overlay_video.py   skeleton trước | sau lên video RGB
      │ wham_refine_mesh.py áp phần sửa vào mesh SMPL → refined/wham_output_refined.pkl
      │ render_wham.py     video kiểu WHAM: bản sửa, so sánh, chồng 2 mesh
      │ change_report.py   bảng "đã sửa gì" (changes.html)
      └ dynamics_report.py inverse dynamics, không sửa góc → ik_exact/dynamics/dynamics.html
```

**Môi trường:**
- `.venv` (WSL, có nimblephysics): `run_cometh.py`, `refine_q.py`, `dynamics_report.py`.
- Env conda `wham` (WSL, có torch, pytorch3d, opencv, joblib): các script còn lại.

**File ngoài cần có:**
- **Model SMPL mà WHAM dùng:** `~/projects/WHAM/dataset/body_models/smpl/SMPL_NEUTRAL.pkl`. Mọi script đọc J_regressor, skin weights và faces từ file này; đổi đường dẫn bằng `--smpl-model`.
  - Cách tải: đăng ký tài khoản tại https://smplify.is.tue.mpg.de, rồi trong `~/projects/WHAM` chạy `bash fetch_demo_data.sh`.
  - Script đó tải `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` và đổi tên thành `SMPL_NEUTRAL.pkl`.
  - Nếu chỉ cần model trung tính, có thể bỏ qua phần tải SMPL nam/nữ và checkpoint.
  - Đọc file cần `chumpy`, đã có sẵn trong env `wham`.
- Repo WHAM tại `~/projects/WHAM`, để dùng renderer.

Kết quả của clip toe loop nằm trong `out/toeloop_wham/`.

## 1. WHAM → CSV keypoint 3D

```bash
python video_pipeline/wham/wham_to_keypoints.py path/to/wham_output.pkl \
    --fps 60 -o video_pipeline/wham/out/x/wham_keypoints.csv
.venv/bin/python video_pipeline/run_cometh.py video_pipeline/wham/out/x/wham_keypoints.csv \
    --out video_pipeline/wham/out/x/ik_exact --ik exact
```

- 24 khớp SMPL được tính bằng `J_regressor @ verts`, sau đó chuyển sang hệ world của WHAM (trục Y hướng lên).
- `--fps`: frame rate của video đã đưa vào WHAM (WHAM không lưu giá trị này).
- `--ground sequence` (mặc định): mặt đất đặt ở phân vị 10% của đỉnh mesh thấp nhất.
- **Lưu ý:** độ cao toàn cục của WHAM không đáng tin. Trên clip toe loop, WHAM giữ người trên mặt đất lúc đang nhảy, rồi để người lơ lửng 7–18 cm sau khi tiếp đất.
- Nên dùng `--ik exact`. Với cú nhảy, `--ik qp` lệch keypoint khoảng 29 cm.

## 2. Kiểm tra / sửa góc khớp: `refine_q.py`

```bash
.venv/bin/python video_pipeline/wham/refine_q.py video_pipeline/wham/out/x/ik_exact \
    --keypoints video_pipeline/wham/out/x/wham_keypoints.csv
```

| Cờ | Ý nghĩa |
|---|---|
| `out_of_rom` | Ngoài biên độ khớp (ROM). ROM mặc định lấy từ `COMETH/parameters.py`. Muốn đổi thì sửa `rom_used.csv` rồi truyền lại bằng `--rom` |
| `at_limit` | Dính sát biên ROM, tức IK muốn đi xa hơn. Mặc định không xét cột sống và xương bả vai, vì COMETH cố ý khoá các khớp này trong ±5–10° (`--no-limit-check`) |
| `spike` | Lệch so với median 5 frame quá `--spike-deg` (mặc định 10°) |
| `too_fast` | Vận tốc góc vượt `--max-vel` (mặc định 1000°/s) |
| `bad_fit` | Cả frame có MPJPE > `--max-mpjpe` (mặc định 50 mm) |

**Cách sửa** (hậu xử lý, không chạy lại IK):
1. Nội suy PCHIP lại các mẫu `spike` / `too_fast`, chỉ với đoạn lỗi dài tối đa `--max-gap` frame.
2. Lọc Butterworth hai chiều `--cutoff` (mặc định 12 Hz) trên các khớp. 6 DOF gốc chỉ được lọc khi bật `--filter-root`.
3. Kẹp giá trị vào ROM.

Riêng `at_limit` và `bad_fit` chỉ được báo, không sửa được bằng cách này. Làm mượt từng DOF riêng lẻ sẽ khiến mô hình lệch keypoint thêm một chút: trên clip toe loop, MPJPE tăng từ 21.6 lên 27.2 mm.

**Output trong `<ik_dir>/refined/`:**
- `q_refined.csv`, `flags.csv`, `rom_used.csv`
- `report.html`: bảng góc khớp, ô lỗi được tô màu, có thể chuyển giữa giá trị trước/sau khi sửa
- `joint_angles_refined.png`
- `joints_before.csv` / `joints_refined.csv`

## 3. Skeleton trước | sau trên video RGB: `overlay_video.py`

```bash
python video_pipeline/wham/overlay_video.py path/to/wham_output.pkl --video original.mp4 \
    --j-regressor path/to/J_regressor_SMPL_MALE.pkl --refined video_pipeline/wham/out/x/ik_exact/refined
```

Output là `refined/compare.mp4`:
- **Skeleton:** bên trái là trước khi sửa (xanh dương), bên phải là sau khi sửa (xanh lá). Skeleton trắng mảnh là input WHAM.
- **Vòng tròn đánh dấu khớp lỗi:** đỏ = ngoài ROM, cam = đột biến / quá nhanh, vàng = dính biên.
- **Tuỳ chọn:** `--layout overlay` để vẽ hai skeleton trên cùng một hình, `--crop 0` để giữ nguyên khung hình.

## 4. Áp phần sửa vào mesh SMPL và render video kiểu WHAM

```bash
python video_pipeline/wham/wham_refine_mesh.py path/to/wham_output.pkl \
    --refined video_pipeline/wham/out/x/ik_exact/refined
python video_pipeline/wham/render_wham.py video_pipeline/wham/out/x/ik_exact/refined/wham_output_refined.pkl \
    --video wham_input.mp4 \
    --compare-pkl path/to/wham_output.pkl --flags video_pipeline/wham/out/x/ik_exact/refined/flags.csv
python video_pipeline/wham/change_report.py --refined video_pipeline/wham/out/x/ik_exact/refined \
    --wham-pkl path/to/wham_output.pkl
```

**Cách áp vào mesh:**
- Mục tiêu của 12 keypoint SMPL = keypoint COMETH đã sửa + độ lệch SMPL–BSM (đã lọc thông thấp `--offset-cutoff`).
- Chỉ fit lại body pose của 23 khớp SMPL. Hướng gốc, vị trí và shape giữ nguyên như WHAM.
- Mesh được biến dạng *tương đối* so với mesh WHAM gốc (LBS với phép biến đổi new·orig⁻¹ và skin weights của SMPL_NEUTRAL), nên shape giữ đúng như WHAM. Không áp lại pose blend shapes.

**Video (`render_wham.py`, chọn bằng `--outputs`):**
- `wham_refined.mp4`: giống video WHAM, nhưng là mesh đã sửa.
- `wham_compare.mp4`: trên là WHAM gốc, dưới là đã sửa. Hai hàng dùng chung mặt đất và camera.
- `wham_overlay.mp4`: hai mesh chồng nhau, xám là gốc, cam là đã sửa (đổi màu bằng `--overlay-color`). Thêm `--heat 30` để tô theo độ lệch.

**Bảng `changes.html`:** gồm 3 tab:
- Δ góc khớp BSM, ô bị gắn cờ có viền vàng;
- góc xoay thay đổi của từng khớp SMPL;
- dịch chuyển mesh theo bộ phận (mm).

Số liệu tương ứng nằm trong `changes_*.csv`.

## 5. Inverse dynamics (không sửa góc) và bảng đánh dấu: `dynamics_report.py`

```bash
.venv/bin/python video_pipeline/wham/dynamics_report.py video_pipeline/wham/out/x/ik_exact \
    --keypoints video_pipeline/wham/out/x/wham_keypoints.csv --flight 12-48
```

**Cách tính:**
- Inverse dynamics cổ điển bằng nimblephysics. Phương trình: M·q̈ + h = [r; τ] + Jcᵀ·f.
- Lực tiếp xúc f nằm trong nón ma sát và được chọn sao cho lực/mô-men dư r ở khung chậu nhỏ nhất. Từ đó suy ra mô-men khớp τ, không bị kẹp.
- Góc khớp được lọc `--cutoff` (mặc định 6 Hz) chỉ để lấy đạo hàm. Bản thân góc không bị sửa.

**Tiếp đất:**
- Pha bay khai báo bằng `--flight`, với frame id đọc từ video, vì độ cao của WHAM không đáng tin.
- Ngoài pha bay, chân thấp hơn được coi là đang chạm đất.
- Kiểm tra: ở tư thế tĩnh, chạm hai chân, kết quả là lực dư ≈ 0 và lực đất = 1.00 × trọng lượng.

**Cờ được đặt khi:**
- mô-men vượt ngưỡng tham khảo theo khớp (`TORQUE_MAX`, Nm/kg);
- lực đất dọc > `--max-grf` × trọng lượng;
- lực dư > 10% trọng lượng;
- mô-men dư > 5% trọng lượng × 1 m.

**Output trong `<ik_dir>/dynamics/`:**
- `dynamics.html`
- `tau.csv`, `grf_residual.csv`, `flags_dynamics.csv`, `dynamics.png`

**Kết quả trên clip toe loop:** lực dư khoảng 210% trọng lượng lúc bay và khoảng 80% lúc trên băng. Nguyên nhân là quỹ đạo toàn cục của WHAM không hợp lý về vật lý, nên mô-men thu được chỉ để phát hiện chỗ dữ liệu sai, chưa phải lực thật của vận động viên.
