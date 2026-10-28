# AI Smart Dustbin — Automated Medical & Biomedical Waste Segregation

An embedded IoT computer vision system designed for the **Raspberry Pi 4** that automates the classification and physical segregation of medical and biomedical waste using a custom Convolutional Neural Network (TensorFlow/Keras) and dual-servo PWM actuation.

---

## 📁 Repository Structure

- **[train.py](train.py)**: Lightweight CNN training script with data augmentation, early stopping, and metric evaluation.
- **[test.py](test.py)**: Real-time PC/desktop webcam testing script with live probability overlay.
- **[stest.py](stest.py)**: Production Raspberry Pi deployment script with camera capture, 2-state FSM, and dual-servo control (`RPi.GPIO`).
- **[SSS.py](SSS.py)**: Hardware servo calibration and pin testing utility.
- **[working.pdf](working.pdf)**: Complete technical specification, pinout diagrams, and FSM documentation.
- **[REFERENCE_DATASETS.md](REFERENCE_DATASETS.md)**: Catalog of 10,000+ public reference images across Kaggle, Roboflow, and Mendeley.
- **[download_reference_images.py](download_reference_images.py)**: Standalone multithreaded reference image fetcher.
- **[dataset/](dataset/)**: Active local training (390 images) and test (96 images) dataset.
- **[class_names.txt](class_names.txt)**: Class mapping (`0:biomedical-waste`, `1:medical_waste`, `2:normal`).

---

## 🏷️ Waste Classes

| ID | Class Name | Description | Servo 2 Chute Angle | Gate (Servo 1) Action |
| :-: | :--- | :--- | :-: | :-: |
| `0` | **biomedical-waste** | Syringes, blood vials, swabs, specimen containers | **50°** | Opens (0°) $\rightarrow$ Closes (100°) |
| `1` | **medical_waste** | Surgical masks, gloves, plastic blister packs | **150°** | Opens (0°) $\rightarrow$ Closes (100°) |
| `2` | **normal** | Clean/empty hopper or background | **100°** (Neutral) | **Locked Closed (100°)** |

---

## 📊 Local Dataset Breakdown

- **Active Training Set** (`dataset/train/`): 390 images (130 biomedical, 130 medical, 130 normal)
- **Active Test Set** (`dataset/test/`): 96 images (32 biomedical, 32 medical, 32 normal)
- **Reference Images** (`dataset/reference_images/`): **2,239 real-world images**
  - `biomedical-waste`: 522 images (gauzes, test tubes, urine bags)
  - `medical_waste`: 1,585 images (latex/nitrile gloves, caps, glasses, shoe covers, filters)
  - `normal`: 132 images (empty hopper/clean background)
- **Total Local Images**: **2,725 images**

---

## 🌐 10,000+ Reference Datasets (Non-Training)

To review or download 10,000+ reference images without affecting the local training set, see **[REFERENCE_DATASETS.md](REFERENCE_DATASETS.md)**.

### Fast Reference Download Options:

1. **Via Kaggle CLI (Single-Step 12k+ archive)**:
   ```bash
   kaggle datasets download -d manasbagale/medical-waste-dataset -p reference_images --unzip
   ```

2. **Via Included Downloader Script**:
   ```bash
   # Download reference images into reference_images/ folder
   python download_reference_images.py --limit 100
   ```