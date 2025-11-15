# COVID-19 Diagnosis Predictor: Classification using Clinical Data

## 1. Executive Summary:

This project focuses on building a machine learning classifier to **predict a positive SARS-CoV-2 (COVID-19) test result** using a sparse, highly unbalanced dataset of routine blood and viral test results collected from patients upon hospital arrival.

The goal is to provide a fast, non-PCR-based diagnostic signal to assist clinical decision-making.

| Metric | Best Model | Result |
| :--- | :--- | :--- |
| **Model** | **Support Vector Machine (SVM)** | RBF Kernel |
| **Final Test F1-Score (Positive Class)** | **0.615** | Prioritizing a balance between Precision and Recall. |
| **Key Challenge** | **High Class Imbalance (9.9% Positive)** | Addressed via Stratified Sampling and F1-Score optimization. |

***

## 2. My Data Science Work Method:

This project followed a clear, iterative process to transform raw data into a informative model:

### **A. Problem Definition & Data Acquisition**
* **Objective:** Binary classification of SARS-CoV-2 test results.
* **Source:** Kaggle dataset: `einsteindata4u/covid19` (Hospital Israelita Albert Einstein).

### **B. Exploratory Data Analysis (EDA) & Cleaning**
* Identify data quality issues (high sparsity, missing values).
* Define the target variable and measure class imbalance.

### **C. Data Preprocessing & Feature Engineering**
* Implement robust pipelines for handling missing values and categorical encoding.
* Create new features to improve model performance.
* **Actionable Step:** Use **Stratified Sampling** during Train/Test split to preserve the rare positive class distribution.

### **D. Modeling & Experiment Tracking (MLFlow)**
* Establish baseline performance across multiple algorithms (RandomForest, GBM, SVM).
* Use **MLFlow** to log all model parameters, metrics, and artifacts for reproducibility.
* Optimize the best-performing model using `RandomizedSearchCV`.

### **E. Evaluation & Deployment Readiness**
* Assess final performance using the **F1-score**, focusing on the ability to correctly identify positive cases (Recall) without excessive false alarms (Precision).

***

## 3. Project Breakdown and Key Learning

### 3.1. Phase 1: Exploratory Data Analysis (EDA)
* **Initial Data Size:** 5644 rows, 111 columns.
* **Sparsity Challenge:** Over 70% of columns had more than 90% missing values. Aggressively dropped these sparse features.
* **Final Feature Set:** Focused on a subset of blood and viral tests (approx. 30 features) with relatively lower missing rates.

### 3.2. Phase 2: Data Preprocessing
* **Feature Engineering:** Created a binary feature, **`sick`**, indicating if a patient tested positive for *any* non-COVID-19 viral test in the panel.
* **Imputation:** For the final selected features, rows with any remaining missing values were dropped to maintain integrity.

### 3.3. Phase 3: Modeling and Results

#### **Model Optimization**
The **Support Vector Machine (SVM)** model was fine-tuned using `RandomizedSearchCV` on a robust pipeline.

* **Final Pipeline Configuration:**
    * Polynomial Features (Degree **3**)
    * Feature Selection (`SelectKBest` = **30 features**)
    * SVM Classifier (`kernel='rbf'`, `C=50`)

* **Final Performance:** An optimal F1-score of **0.615** was achieved on the unseen test set after **threshold adjustment** (`decision_function(X) > -1`), which was crucial for balancing Precision and Recall in this imbalanced classification task.

***

## 4. Experiment Tracking with MLFlow

**MLFlow** was integrated as the central hub for experiment management.
* **Logged Experiments:** All baseline models and `RandomizedSearchCV` runs were tracked.
* **Logged Artifacts:** Model parameters, final metrics (F1-score, Precision, Recall), and the serialized Python model object were saved.
* **Value-Add:** Provides a clear audit trail for comparing model iterations and transitioning the final model into a registry for deployment.

***

## 5. Getting Started (Setup & Run)

**Run the Analysis:** Read the following notebooks sequentially:
  * `Diagnosis_of_COVID_19_and_its_clinical_spectrum_(EDA).ipynb`
  * `Diagnosis_of_COVID_19_and_its_clinical_spectrum_(DATA_PROCESSING).ipynb`
  * `Diagnosis_of_COVID_19_and_its_clinical_spectrum_(MODELING).ipynb`

***

## 6. Data Source & Acknowledgements

* **Source:** [Kaggle: COVID-19 Clinical Data](https://www.kaggle.com/einsteindata4u/covid19) (Hospital Israelita Albert Einstein, São Paulo, Brazil).
