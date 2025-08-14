# EZEnhance — Photo Colorizer & Enhancer

Drag‑and‑drop old photo restoration with a modern, flat UI. Restore, colorize, compare, and save your photos — all locally.

> Built with **PySide6**, **OpenCV**, **NumPy**, and **Pillow**.

---

## ✨ Features

* **Drag & drop** images (or use **Open**)
* **Auto Restore** pipeline: denoise → dust/scratch reduction → CLAHE contrast → unsharp mask → optional upscale (1.5× / 2×)
* **Smart Colorization**

  * Uses **AI colorization** automatically if model files are present
  * Falls back to a tasteful **smart tint** approach when AI isn’t available
* **Fine‑tune controls**: Denoise, Dust/Scratch, Contrast, Sharpen, Upscale, Strength, Vibrance, Warmth, AI toggle
* **Compare view**: side‑by‑side Original ↔ Result with a neon divider
* **Undo** history (up to \~20 steps)
* **Modern UI**: green/gray theme, rounded corners, emoji hints
* **Local‑only** processing (no cloud uploads)

---

## 🧩 Requirements

* **Python**: 3.9+ (3.10/3.11 recommended)
* **OS**: Windows, macOS, or Linux
* **Dependencies**:

  * `PySide6`
  * `opencv-python`
  * `numpy`
  * `pillow`

Install them with:

```bash
python -m pip install --upgrade pip
pip install PySide6 opencv-python numpy pillow
```

> 💡 If you plan to use AI colorization, you’ll also need the model files below.

---

## 🤖 Optional: AI Colorization Models

EZEnhance auto‑detects the OpenCV colorization model when these files are placed **next to the script**:

* `colorization_deploy_v2.prototxt`
* `colorization_release_v2.caffemodel`
* `pts_in_hull.npy`

If present, AI colorization is used for grayscale images; otherwise EZEnhance uses a smart tint fallback. File names must match exactly.

> Tip: Keep the three files in the same folder as `ezenhance.py`.

---

## 📦 Installation (recommended via virtual env)

```bash
# 1) Create & activate a virtual environment (Windows)
python -m venv .venv
.venv\\Scripts\\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install PySide6 opencv-python numpy pillow
```

Optionally, place the AI model files in the same directory as the script (see above).

---

## 🚀 Run

```bash
python ezenhance.py
```

On first launch you’ll see a welcoming drop area. Drag a photo in or click **Open**.

---

## 🖱️ How to Use

1. **Open** a photo (JPEG/PNG/TIFF/BMP) or drag & drop it into the preview.
2. Use **🧼 Restore** controls to gently clean and sharpen your image.
3. Use **🎨 Colorization** controls:

   * **Strength**: blends colorized output with the original
   * **Vibrance**: boosts low‑saturation regions more than already‑vivid ones
   * **Warmth**: nudges tones warmer/cooler
   * **Use AI**: toggles AI colorization if models are available
4. Click **✨ Restore + Colorize** for a one‑click workflow.
5. Use **↔ Compare** to view Original vs Result.
6. **↩ Undo** reverts the last step (history depth \~20).
7. **💾 Save** exports your result (PNG/JPEG).

---

## 🖼️ Supported Formats

* Input: PNG, JPG/JPEG, TIFF, BMP (others may work via PIL fallback)
* Output: PNG, JPEG

---

## ⚙️ Settings Reference

**Restoration**

* *Denoise* (0–30): Non‑local means noise reduction
* *Dust/Scratch* (0–30): Median blur + morphological opening (edge‑preserving blend)
* *Contrast* (0–100): CLAHE strength on L channel (LAB)
* *Sharpen* (0–50): Edge‑preserving unsharp mask
* *Upscale* (0/1/2): 0 = none, 1 = 1.5×, 2 = 2× (Lanczos)

**Colorization**

* *Strength* (0–100): Mix colorized result with original
* *Vibrance* (0–100): Preferential saturation boost for flatter areas
* *Warmth* (\~−50…+50): Temperature bias (UI slider centered at 50)
* *Use AI if available*: Enable OpenCV‑DNN Caffe model when present

---

## 🔍 How It Works (High Level)

* **Restoration**

  * *Denoise*: Gentle NL‑Means to reduce chroma/luma noise
  * *Dust/Scratch*: Median filtering + morphological opening, softly blended
  * *Contrast*: CLAHE on L (LAB space) for controlled local contrast
  * *Sharpen*: Gaussian‑based unsharp mask with tuned amount
  * *Upscale*: High‑quality Lanczos interpolation
* **Colorization**

  * *AI path*: OpenCV DNN with Caffe model predicts `ab` chroma for the `L` channel
  * *Fallback path*: Generates subtle pseudo‑chroma from luminance structure, with edge‑aware vibrance and warmth bias, then blends by *Strength*

---

## 🧰 Troubleshooting

* **"Could not read image"**: Ensure the file exists and is a supported format. Try opening it in another app; if it’s unusual, convert to PNG/JPEG and retry.
* **No window / Qt plugin error**: Reinstall PySide6; on Linux ensure desktop libs are present (e.g., `libxcb`).
* **AI doesn’t trigger**: Confirm all three model files exist in the script folder with exact names.
* **High‑DPI blurry UI**: The app enables High‑DPI pixmaps; check OS scaling settings.
* **Performance**: Large scans can be heavy. Try processing at native size first; use Upscale only when needed.

---

## 🧪 Packaging (optional)

Create a single‑file executable with **PyInstaller**:

```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed \
  --name EZEnhance \
  --add-data "colorization_deploy_v2.prototxt:." \
  --add-data "colorization_release_v2.caffemodel:." \
  --add-data "pts_in_hull.npy:." \
  ezenhance.py
```

The `--add-data` entries are optional; include them if you’re bundling the AI models.

---

## 🔐 Privacy

All processing happens **locally on your machine**. Images are never uploaded.

---

## 📄 License

**MIT License** — see header in source file.

---

## 🙌 Credits

* OpenCV team for the colorization DNN (Caffe) architecture and reference model.
* Qt/PySide6 for the UI framework.

---

## 💬 Support & Contributions

Issues and PRs welcome. If you spot a bug or want a feature (e.g., crop, scratch inpainting, batch mode), open a ticket or contribute!
