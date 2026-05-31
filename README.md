# 🌋 Earthquake Indonesia — Workflow CI

Repository ini merupakan bagian dari submission **Kriteria 3: Membuat Workflow CI** pada kelas Machine Learning System.
Pipeline ini melakukan re-training model machine learning secara otomatis menggunakan **MLflow Project** dan **GitHub Actions** setiap kali trigger dipantik.

---

## 📌 Informasi Proyek

| Item | Detail |
|------|--------|
| **Dataset** | USGS Earthquake Database (filter: Indonesia) |
| **Task** | Binary Classification |
| **Target Variable** | `significant` → magnitude ≥ 5.0 = 1, magnitude < 5.0 = 0 |
| **Model** | RandomForestClassifier (Scikit-Learn) |
| **MLflow Version** | 2.19.0 |
| **Python Version** | 3.12.7 |
| **CI Platform** | GitHub Actions |
| **Artifact Storage** | GitHub Actions Artifacts |

---

## 📁 Struktur Repository

```
earthquake_workflow_ci/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions workflow
└── MLProject/
    ├── MLProject                           # Konfigurasi MLflow Project
    ├── conda.yaml                          # Environment dependencies
    ├── modelling.py                        # Script training model
    ├── TautanDkriteria.txt                 # Tautan Docker Hub (Advanced)
    └── earthquake_indonesia_preprocessing/ # Dataset siap latih
        └── earthquake_indonesia_preprocessing.csv
```

---

## ⚙️ Cara Kerja CI Pipeline

Workflow CI terpantik secara otomatis ketika:
- **Push** ke branch `main`
- **Pull request** ke branch `main`
- **Manual trigger** via GitHub Actions → Run workflow

Setelah terpantik, pipeline menjalankan tahapan berikut:

```
Set up job
    ↓
Run actions/checkout@v3
    ↓
Set up Python 3.12.7
    ↓
Check Env
    ↓
Install dependencies
    ↓
Set MLflow Tracking URI
    ↓
Run mlflow project
    ↓
Get latest MLflow run_id
    ↓
Install Python dependencies
    ↓
Upload to GitHub (Artifacts)
```

---

## 🚀 Cara Menjalankan Secara Lokal

### 1. Clone repository

```bash
git clone https://github.com/IsaMusawi/earthquake_workflow_ci.git
cd earthquake_workflow_ci
```

### 2. Install dependencies

```bash
pip install mlflow==2.19.0 scikit-learn==1.6.0 pandas numpy geopy matplotlib seaborn
```

### 3. Jalankan MLflow Project

```bash
mlflow run MLProject/ --env-manager=local \
  -P data=MLProject/earthquake_indonesia_preprocessing/earthquake_indonesia_preprocessing.csv
```

### 4. Lihat hasil di MLflow UI

```bash
mlflow ui --port 5000
# Buka browser: http://127.0.0.1:5000
```

---

## 📊 MLflow Project Configuration

File `MLProject` mendefinisikan entry point dan environment:

```yaml
name: earthquake-indonesia-classification

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data:
        type: str
        default: "earthquake_indonesia_preprocessing/earthquake_indonesia_preprocessing.csv"
    command: "python modelling.py --data {data}"
```

---

## 🔧 Environment Dependencies (`conda.yaml`)

```yaml
name: mlflow-earthquake
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.12.7
  - pip
  - pip:
    - mlflow==2.19.0
    - scikit-learn==1.6.0
    - pandas==2.2.3
    - numpy==2.2.1
    - geopy==2.4.1
    - matplotlib==3.10.0
    - seaborn==0.13.2
```

---

## 📈 Fitur Model

| Fitur | Deskripsi |
|-------|-----------|
| `Latitude` | Koordinat lintang episentrum |
| `Longitude` | Koordinat bujur episentrum |
| `Depth` | Kedalaman gempa (km) |
| `Magnitude` | Kekuatan gempa |
| `depth_category` | Kategori kedalaman: shallow / intermediate / deep |
| `year` | Tahun kejadian |
| `month` | Bulan kejadian |
| `day_of_week` | Hari dalam minggu |
| `decade` | Dekade kejadian |

**Target:** `significant` — 1 jika magnitude ≥ 5.0, 0 jika sebaliknya

---

## 🗂️ Artefak yang Disimpan

Setiap kali CI berhasil berjalan, artefak berikut disimpan ke **GitHub Actions Artifacts**:

- Model terlatih (`model.pkl` / MLflow model)
- Metrik evaluasi (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix
- Model signature & input examples
- `MLmodel`, `conda.yaml`, `python_env.yaml`, `requirements.txt`

Artefak dapat diunduh dari tab **Actions → pilih run → Artifacts**.

---

## 🔗 Tautan Terkait

| Resource | Link |
|----------|------|
| Repository Kriteria 1 (Eksperimen) | [Eksperimen_Muhammad-Isa-Almusawi](https://github.com/IsaMusawi/earthquake_sml.git) |
| Dataset Sumber | [USGS Earthquake Database — Kaggle](https://www.kaggle.com/datasets/usgs/earthquake-database) |

---

## 👤 Author

**Muhammad Isa Almusawi**
Submission SMSML — Dicoding Indonesia