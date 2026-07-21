# 🍷 Wine Quality Prediction using Decision Tree

A Machine Learning project that predicts whether a **red wine** is **Good** or **Bad** based on its physicochemical properties. The project compares multiple classification algorithms and selects the best-performing model using **GridSearchCV**.

---

## 📌 Project Overview

The objective of this project is to classify wine quality into two categories:

- **GOOD** → Quality score **≥ 7**
- **BAD** → Quality score **< 7**

The dataset is preprocessed, analyzed, and used to train multiple machine learning models. Hyperparameter tuning is performed to improve the final model's performance.

## 🚀 Live Demo

🔗 https://wine-quality-prediction-4udjvczvbtzwapmgvdtmgn.streamlit.app/
---

## 📂 Dataset

- **Dataset:** Wine Quality Dataset (Red Wine)
- **Number of Samples:** 1,599
- **Features:** 11 Chemical Properties
- **Target:** Wine Quality

### Features

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

Target:

- Quality (Converted into GOOD/BAD)

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📊 Exploratory Data Analysis

The project includes:

- Dataset overview
- Missing value analysis
- Statistical summary
- Correlation analysis
- Feature importance visualization
- Class distribution visualization

### Correlation Highlights

Positive correlation with quality:

- Alcohol
- Sulphates
- Citric Acid

Negative correlation with quality:

- Volatile Acidity
- Total Sulfur Dioxide
- Density

---

## 🤖 Machine Learning Models

The following models were trained and compared:

### Logistic Regression

| Metric | Score |
|---------|--------|
| Accuracy | 89.38% |
| Precision | 69.57% |
| Recall | 37.21% |
| F1 Score | 48.48% |

---

### K-Nearest Neighbors (KNN)

| Metric | Score |
|---------|--------|
| Accuracy | 89.38% |
| Precision | 66.67% |
| Recall | 41.86% |
| F1 Score | 51.43% |

---

### Decision Tree

| Metric | Score |
|---------|--------|
| Accuracy | 90.00% |
| Precision | 61.70% |
| Recall | 67.44% |
| F1 Score | 64.44% |

---

## 🚀 Hyperparameter Tuning

GridSearchCV was used with 5-fold cross-validation.

### Best Parameters

```python
{
    'criterion': 'entropy',
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'class_weight': None
}
```

---

## 🏆 Final Model Performance

### Tuned Decision Tree

| Metric | Score |
|---------|--------|
| Accuracy | **92.50%** |
| Precision | **73.17%** |
| Recall | **69.77%** |
| F1 Score | **71.43%** |

### Confusion Matrix

```
[[266  11]
 [ 13  30]]
```

---

## 📈 Feature Importance

The most influential features for predicting wine quality are:

| Feature | Importance |
|---------|------------|
| Alcohol | 0.2859 |
| Sulphates | 0.1017 |
| Volatile Acidity | 0.0973 |
| pH | 0.0866 |
| Total Sulfur Dioxide | 0.0789 |
| Chlorides | 0.0717 |
| Residual Sugar | 0.0714 |
| Citric Acid | 0.0614 |
| Free Sulfur Dioxide | 0.0581 |
| Density | 0.0533 |
| Fixed Acidity | 0.0338 |

---

## 🔮 Prediction Example

Input:

```
Fixed Acidity: 7.4
Volatile Acidity: 0.35
Citric Acid: 0.46
Residual Sugar: 2.1
Chlorides: 0.06
Free Sulfur Dioxide: 15
Total Sulfur Dioxide: 45
Density: 0.995
pH: 3.35
Sulphates: 0.72
Alcohol: 11.5
```

Output:

```
Predicted Quality : GOOD
Confidence : 92%
```

---

## 📁 Project Structure

```
Wine-Quality-Prediction/
│
├── winequality.csv
├── Wine_Quality_Prediction.ipynb
├── README.md
├── requirements.txt
└── images/
    ├── correlation_heatmap.png
    ├── feature_importance.png
    └── confusion_matrix.png
```

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/Wine-Quality-Prediction.git
```

### Move into the project folder

```bash
cd Wine-Quality-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook
```

Open:

```
Wine_Quality_Prediction.ipynb
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📌 Key Learnings

- Data preprocessing
- Binary classification
- Feature engineering
- Exploratory Data Analysis (EDA)
- Logistic Regression
- KNN Classification
- Decision Tree Classification
- Hyperparameter tuning using GridSearchCV
- Model evaluation using Accuracy, Precision, Recall, and F1-score
- Feature Importance Analysis

---

## 📈 Future Improvements

- Random Forest Classifier
- XGBoost
- LightGBM
- CatBoost
- SMOTE for handling class imbalance
- Flask/FastAPI web deployment
- Streamlit web application
- Docker deployment

---

## 👨‍💻 Author

**Luvkush Singh**

B.Tech CSE (AI & ML)  
VIT Bhopal University

---

## ⭐ If you found this project helpful, don't forget to star the repository!
