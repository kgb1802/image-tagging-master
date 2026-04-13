# 🖼️ Automatic Image Tagging using Deep Learning

Auto-tag images using three different pretrained deep learning models — Google's OpenImages pretrained model, InceptionV3, and ResNet50 — all integrated into a single unified pipeline.

---

## 📌 Project Overview

This project implements an **AutoTagEngine** that automatically generates descriptive tags for images using pretrained CNN models. It combines three approaches:

| Script | Model | Dataset | Output |
|---|---|---|---|
| `tag_openimage_tf.py` | ResNet-101 (TF v1.x) | Google Open Images V2 | `auto_tag_result.csv` |
| `tag_inception_keras.py` | InceptionV3 (Keras) | ImageNet | Console + JSON |
| `tag_resnet_keras.py` | ResNet50 (Keras) | ImageNet | Console + JSON |

---

## 📂 Project Structure

```
auto-image-tagging/
├── pretrain_open_images/        # OpenImages pretrained model files (download separately)
│   ├── oidv2-resnet_v1_101.ckpt
│   ├── classes-trainable.txt
│   └── class-descriptions.csv
├── test_images/                 # Place your test images here (.jpg / .JPG)
├── tag_openimage_tf.py          # Method 1 — OpenImages ResNet-101
├── tag_inception_keras.py       # Method 2 — InceptionV3 (Keras)
├── tag_resnet_keras.py          # Method 3 — ResNet50 (Keras)
├── auto_tag_result.csv          # Output from Method 1
├── result.json                  # Output from Method 1 (JSON format)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.6
- `pip`
- `virtualenv`

### 1. Create and activate virtual environment

```bash
virtualenv venv -p python3.6
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **SSL Issue?** If you encounter SSL errors during pip install, update pip first:
> ```bash
> curl https://bootstrap.pypa.io/get-pip.py | python
> ```

### 3. Download the OpenImages pretrained model

Download the pretrained model (V2) from the link below and unzip all files into the `pretrain_open_images/` folder:

🔗 [Download OpenImages Pretrained Model V2](https://storage.googleapis.com/openimages/2017_07/oidv2-resnet_v1_101.ckpt.zip)

Your folder should look like:
```
pretrain_open_images/
├── oidv2-resnet_v1_101.ckpt.data-00000-of-00001
├── oidv2-resnet_v1_101.ckpt.index
├── oidv2-resnet_v1_101.ckpt.meta
├── classes-trainable.txt
└── class-descriptions.csv
```

---

## 🚀 Running the Taggers

> Make sure your virtual environment is active before running any script.

### Method 1 — Google OpenImages ResNet-101

```bash
python tag_openimage_tf.py
```
Results saved to `auto_tag_result.csv` and `result.json`

---

### Method 2 — InceptionV3 (Keras in TensorFlow)

```bash
python tag_inception_keras.py
```

---

### Method 3 — ResNet50 (Keras in TensorFlow)

```bash
python tag_resnet_keras.py
```

---

## 📊 Sample Results

Test image: `bike.JPG`

**InceptionV3:**
```
Predicted: [
  ('n03792782', 'mountain_bike',          0.8788),
  ('n03208938', 'disk_brake',             0.0120),
  ('n02835271', 'bicycle-built-for-two',  0.0063),
  ('n03649909', 'lawn_mower',             0.0028),
  ('n03127747', 'crash_helmet',           0.0021)
]
```

**ResNet50:**
```
Predicted: [
  ('n03792782', 'mountain_bike',          0.6810),
  ('n02835271', 'bicycle-built-for-two',  0.3081),
  ('n04482393', 'tricycle',               0.0053),
  ('n03127747', 'crash_helmet',           0.0013),
  ('n03649909', 'lawn_mower',             0.0009)
]
```

Both models correctly identify `mountain_bike` as the top prediction. InceptionV3 shows higher confidence (0.88) compared to ResNet50 (0.68) on this image.

---

## 🧠 Models Used

### 1. ResNet-101 — Google OpenImages (Method 1)
- Pretrained on **Google Open Images V2** (~9M images, **5000+ categories**)
- Supports **multi-label classification** — multiple tags per image
- Uses **TensorFlow v1.x** session-based inference
- Outputs ranked tags with confidence scores

### 2. InceptionV3 — Keras (Method 2)
- Pretrained on **ImageNet** (~1.2M images, **1000 categories**)
- Loaded via `keras.applications.InceptionV3`
- Good for general object recognition

### 3. ResNet50 — Keras (Method 3)
- Pretrained on **ImageNet** (~1.2M images, **1000 categories**)
- Loaded via `keras.applications.ResNet50`
- Residual connections for stable deep network training

---

## 🔧 Updates from Original Source

This project is based on the following references, with the following updates applied:

- [hahv/auto-tagging-imag-pretrained-model](https://github.com/hahv/auto-tagging-imag-pretrained-model) — Method 1 (OpenImages)
- [karthikmswamy/Keras_In_TensorFlow](https://github.com/karthikmswamy/Keras_In_TensorFlow) — Methods 2 & 3
- [Towards Data Science article](https://towardsdatascience.com/image-tagging-with-keras-in-tensorflow-1-2-bc43c1058019) — Reference for Methods 2 & 3

**Changes made:**
- Upgraded to **Python 3.6** (original used Python 2.7)
- Updated all packages to compatible versions in `requirements.txt`
- Unified all three methods under a single virtual environment
- Fixed deprecated TensorFlow API calls for newer compatibility
- Updated instructions for cross-platform setup (Linux/macOS/Windows)

---

## ⚡ GPU Acceleration (Optional)

This project runs on **CPU by default**. To enable GPU acceleration (significantly faster):

1. Install CUDA and cuDNN compatible with your TensorFlow version
2. Replace `tensorflow` with `tensorflow-gpu` in `requirements.txt`
3. Follow the official [TensorFlow GPU installation guide](https://www.tensorflow.org/install/gpu)

---

## 🔭 Future Scope

- Upgrade to TensorFlow 2.x / Keras eager execution mode
- Add web API (FastAPI/Flask) for real-time tagging
- Extend to video frame-by-frame tagging
- Integrate object localization (bounding boxes)
- Deploy on cloud (GCP Vision / AWS Rekognition hybrid)

---

## 👤 Author

**Kumar Gourav Behera**
B.Tech — Computer Science & Engineering (IoT & IS)
Manipal University Jaipur

---

## 📝 Changelog

```
docs(readme): overhaul README for clarity and completeness

- Rewrote project overview with method comparison table
- Added step-by-step virtualenv setup and SSL fix note
- Documented all 3 models (OpenImages ResNet-101, InceptionV3, ResNet50)
- Included sample predictions for bike.JPG with confidence scores
- Added source references, GPU notes, and future scope
```

---

## 📄 License

This project is for educational and research purposes.
Original model weights belong to their respective owners (Google OpenImages, ImageNet).
