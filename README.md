# Intro_to_ML_PCyay - Classifying Myocardial Infarction (pMI vs Healthy)

Binary classification of subjects as **healthy** or **post-myocardial infarction (pMI)** using a combination of tabular demographic data and 3D cardiac geometry (point-cloud) data across the cardiac cycle.

Coursework for the CDT in Health Data Science – *Introduction to Machine Learning* (Classification task).

---

## Project overview

The pipeline:

1. **Exploratory data analysis** of the demographic features (age, BMI, height, weight, blood pressure, sex) and the 3D geometry.
2. **Quality control** – identifying and removing anatomically impossible samples from the geometry data.
3. **Feature engineering** – combining demographics, PCA components of the geometry, and hand-crafted motion features.
4. **Model selection** – comparing Logistic Regression, Random Forest and Gradient Boosted Trees using stratified 5-fold cross-validation and grid-search hyper-parameter tuning.
5. **Final evaluation** on a held-out test set.

---

## Data

| Source | Description |
|---|---|
| `data/training_data/mixed_demographics.csv` | 900 subjects: `MI` label (`healthy` / `pMI`), `age`, `BMI`, `height`, `weight`, `diastolic_BP`, `systolic_BP`, `sex`. Roughly balanced (452 healthy / 448 pMI), no missing values. |
| `data/training_data/mixed_samples/*.npy` | One file per subject (e.g. `12pMI.npy`). Array of shape `(10 frames, 18000 points, 4)` – x, y, z coordinates plus a region label for each point, over 10 time frames of the cardiac cycle. |
| `data/training_data/clean_samples/` | Geometry files after removing impossible anatomies. |
| `data/training_data/impossible_samples/` | The 38 subjects flagged as anatomically impossible. |
| `data/test_samples/test.csv` | 100 unlabelled test subjects (demographics only). |

> **Note:** `.npy` geometry files are **not** tracked in git (see `.gitignore`) because of their size. Download them from the course page and place them in the folders above before running the code.

---

## Repository structure

```
Intro_to_ML_PCyay/
├── data/
│   ├── training_data/          # demographics CSV + geometry .npy folders (npy not tracked)
│   └── test_samples/           # unlabelled test demographics
├── notebooks/
│   ├── tabular_data_greggy.ipynb          # EDA of demographic features (box/violin/pair plots, correlations)
│   ├── animated_geom_data.py              # 3D animations of example hearts across the 10 frames
│   ├── PCA_geometric.py                   # PCA of geometry, pairplots coloured by sex / label
│   ├── PCA_geom_indexed.py                # Same PCA with subject indices annotated (to spot outliers)
│   ├── global_PCA.py                      # PCA on summary features of each frame
│   ├── clustering.py                      # PCA + KMeans to isolate impossible anatomies
│   ├── pca_good_only.py                   # PCA on cleaned samples only
│   ├── geometric_data_pipeline.py         # Feature preparation for geometry data
│   └── demographics_classification.ipynb  # MAIN NOTEBOOK: cleaning, features, CV, tuning, test evaluation
├── results/                    # Saved figures and animations
├── tests/                      # Test CSVs (labelled and unlabelled)
├── src/                        # (placeholder for reusable code)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Methods

### 1. Exploratory analysis
- Distribution of each demographic feature by group (box and violin plots), pairwise scatter plots and a correlation heatmap (`results/pairwise_correlations.png`, `results/correlation_heatmap.png`).
- Animated 3D point clouds of sample hearts (`results/10_samples_animation.gif`).
- PCA of the geometry coloured by sex and label (`results/pca_pairplot_sex.png`).

### 2. Removing impossible anatomies
PCA of the geometry revealed a distinct group of outlying samples. These were isolated with PCA (95% variance) + KMeans (k = 2), inspected visually (`results/weird_samples_animation.gif`) and confirmed as anatomically impossible. **38 subjects were removed**, leaving **862** subjects.

### 3. Train / test split
Stratified split of the 862 clean subjects (`random_state=9888`):
- **Development set:** 712 subjects (360 healthy / 352 pMI)
- **Held-out test set:** 150 subjects (76 healthy / 74 pMI)

### 4. Features
Each subject is represented by:
- **Demographics (7):** age, BMI, height, weight, diastolic BP, systolic BP, sex
- **Geometry PCA (7):** the coordinates of all 10 frames plus one-hot encoded region labels are standardised and reduced to 7 principal components (~77% variance retained)
- **Motion features (5):** mean and std of position, mean and std of frame-to-frame displacement, and mean total distance travelled per point

All preprocessing (one-hot encoder, scaler, PCA) is **fit only on the training fold** inside cross-validation to avoid data leakage.

### 5. Models and tuning
Stratified 5-fold cross-validation on the development set, with grid search over:

| Model | Grid |
|---|---|
| Logistic Regression (with standardisation) | `C` ∈ {0.1, 1, 10, 50, 100} |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| Gradient Boosted Trees | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |

Model selection was based on mean cross-validated **ROC AUC**.

---

## Results

### Cross-validation (tuned hyper-parameters, 5 folds)

| Model | ROC AUC | F1 | Best params |
|---|---|---|---|
| Logistic Regression | 0.860 ± 0.013 | 0.762 ± 0.021 | `C=100` |
| Random Forest | 0.907 ± 0.012 | 0.830 ± 0.020 | 500 trees, `max_features=0.5` |
| **Gradient Boosted Trees** | **0.923 ± 0.011** | **0.838 ± 0.014** | 300 trees, `lr=0.3`, `max_depth=5`, `subsample=0.8` |

### Held-out test set (150 subjects)

| Model | Accuracy | F1 (pMI) | ROC AUC |
|---|---|---|---|
| Logistic Regression | 0.77 | 0.77 | 0.844 |
| Random Forest | **0.86** | **0.85** | 0.892 |
| Gradient Boosted Trees | 0.84 | 0.83 | **0.893** |

Tree-based models clearly outperform logistic regression, suggesting non-linear relationships between the features and MI status. Adding geometry and motion features on top of demographics improves discrimination.

---

## How to run

```bash
git clone https://github.com/alicerosewh/Intro_to_ML_PCyay.git
cd Intro_to_ML_PCyay
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Place the `.npy` geometry folders inside `data/training_data/` (see **Data** above).
2. Run the `.py` scripts **from the repository root**, e.g. `python notebooks/clustering.py`.
3. Open and run `notebooks/demographics_classification.ipynb` **from inside the `notebooks/` folder** (it uses `../data/...` paths).

All random seeds are fixed (`random_state=9888`) for reproducibility.

---

## Authors

- Alice Wilson
- *(add other team members)*

## License

See [LICENSE](LICENSE).