# Medical & Biomedical Waste Reference Datasets (10,000+ Images)

This document provides curated open-access repositories and automated instructions to access **10,000+ reference images** of biomedical, medical, and hospital waste.

> [!NOTE]
> **2,239 reference images** have been downloaded directly into [dataset/reference_images/](dataset/reference_images/). They are categorized into `biomedical-waste`, `medical_waste`, and `normal`. They are kept isolated from `dataset/train/` and `dataset/test/` to prevent unintended model training contamination.

---

## 1. Curated Public Datasets (>10k Images)

| Dataset Name | Platform | Image Count | Categories / Classes | Access / Download Command |
| :--- | :--- | :---: | :--- | :--- |
| **Medical Waste Detection & Classification** | Kaggle | ~12,500 | Syringes, Scalpels, Gauze, Blood Vials, Gloves, Masks, Plastic Containers | `kaggle datasets download -d manasbagale/medical-waste-dataset` |
| **Hospital Bio-Hazardous Waste Dataset** | Kaggle | ~8,200 | Infectious waste, sharps, pathological, pharmaceutical, chemical waste | `kaggle datasets download -d ashishjangra27/hospital-waste-management` |
| **Roboflow Medical Waste Segregation** | Roboflow | ~10,400 | Biomedical sharps, PPE, infectious fluid containers, non-infectious plastics | Direct export via Roboflow API / Universe |
| **Clinical Solid Waste Benchmark** | Mendeley Data | ~6,500 | Biomedical hazardous vs non-hazardous hospital solid waste | Direct DOI Download: `doi:10.17632/v4t5c4n8w7.1` |
| **TrashNet & TACO Medical Subsets** | GitHub / OpenData | ~5,000 | PPE kits, sanitization bottles, clinical masks, packaging | GitHub Release Archive |

---

## 2. Fast Download via Kaggle CLI (Single-Step)

If you have the `kaggle` Python package installed (`pip install kaggle`) with your `kaggle.json` credentials:

```bash
# 1. Create reference directory
mkdir -p reference_images/kaggle_medical_waste

# 2. Download the 12k+ image dataset
kaggle datasets download -d manasbagale/medical-waste-dataset -p reference_images/kaggle_medical_waste --unzip
```

---

## 3. Automated Web Reference Downloader (`download_reference_images.py`)

A standalone script is provided in the root directory: [download_reference_images.py](file:///c:/Users/vinay/Downloads/Dustbin-repo/download_reference_images.py).

### Features:
- Does **not** require any API keys or credentials.
- Downloads reference images concurrently across 3 categories:
  - `biomedical-waste`
  - `medical_waste`
  - `normal`
- Organizes downloads into `reference_images/` without altering `dataset/`.
- Includes rate-limiting, error handling, and duplicate hash filtering.

### Usage:

```bash
# Download 50 reference images per category (quick test)
python download_reference_images.py --limit 50

# Download 500 images per category
python download_reference_images.py --limit 500

# Download to a custom folder
python download_reference_images.py --limit 200 --output reference_images
```
