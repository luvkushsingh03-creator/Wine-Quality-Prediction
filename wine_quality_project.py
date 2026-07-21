# ============================================================
# PROJECT 1: WINE QUALITY PREDICTION (Decision Tree Focus)
# Paste each "# --- Cell" section into a separate Colab cell
# ============================================================

# --- Cell 1: Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)

sns.set(style="whitegrid")


# --- Cell 2: Load Dataset from YOUR file path ---
#
# OPTION A — Running LOCALLY (Jupyter/Anaconda/VS Code on your own PC):
# Use a raw string (r"...") so backslashes in the Windows path aren't
# misread as escape characters.
FILE_PATH = r"C:\Users\luvku\Downloads\winequality (1) (1) (1).csv"
#
# OPTION B — Running in GOOGLE COLAB (cloud, can't see your PC's files):
# 1) Run this in a cell first, then choose your file from the upload dialog:
#      from google.colab import files
#      uploaded = files.upload()
# 2) Then set:
#      FILE_PATH = "winequality (1) (1) (1).csv"
#
# Uncomment ONE of the two lines above depending on where you're running this.

df = pd.read_csv(FILE_PATH)   # add sep=";" here if your CSV uses semicolons instead of commas

print(df.head())
print(df.info())
print(df.describe())


# --- Cell 3: Missing Values Check ---
print("Missing values per column:\n", df.isnull().sum())


# --- Cell 4: Correlation Analysis ---
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

print("\nCorrelation with quality:\n", corr["quality"].sort_values(ascending=False))


# --- Cell 5: Create Binary Target ---
df["quality_label"] = df["quality"].apply(lambda q: 1 if q >= 7 else 0)

print(df["quality_label"].value_counts())
sns.countplot(x="quality_label", data=df)
plt.title("Class Distribution (0 = BAD, 1 = GOOD)")
plt.show()


# --- Cell 6: Features and Target ---
FEATURE_COLUMNS = [c for c in df.columns if c not in ["quality", "quality_label"]]

X = df[FEATURE_COLUMNS]
y = df["quality_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Cell 7: Helper function to evaluate any model ---
def evaluate_model(model, X_train, X_test, y_train, y_test, name="Model", plot=True):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BAD", "GOOD"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.show()

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "confusion_matrix": cm}


# --- Cell 8: Logistic Regression WITHOUT Scaling ---
log_reg = LogisticRegression(max_iter=1000, random_state=42)
result_unscaled = evaluate_model(log_reg, X_train, X_test, y_train, y_test,
                                  "Logistic Regression (No Scaling)")


# --- Cell 9: Apply StandardScaler ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Cell 10: Logistic Regression WITH Scaling ---
log_reg_scaled = LogisticRegression(max_iter=1000, random_state=42)
result_scaled = evaluate_model(log_reg_scaled, X_train_scaled, X_test_scaled,
                                y_train, y_test, "Logistic Regression (Scaled)")

print("\nComparison — Logistic Regression Before vs After Scaling:")
print(pd.DataFrame([result_unscaled, result_scaled]).drop(columns="confusion_matrix"))


# --- Cell 11: Compare Three Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = []
for name, model in models.items():
    if name == "Decision Tree":
        res = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    else:
        res = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
    results.append(res)

results_df = pd.DataFrame(results).drop(columns="confusion_matrix").sort_values(by="f1", ascending=False)
print("\n===== Model Comparison (sorted by F1-score) =====")
print(results_df)

best_model_name = results_df.iloc[0]["model"]
print(f"\nBest performing model: {best_model_name}")


# --- Cell 12: Hyperparameter Tuning on Decision Tree using GridSearchCV ---
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"]
}

dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV F1-score:", grid_search.best_score_)

best_dt = grid_search.best_estimator_


# --- Cell 13: Evaluate Tuned Decision Tree on Test Set (FINAL MODEL) ---
FINAL_MODEL_NAME = "Decision Tree (Tuned with GridSearchCV)"
final_result = evaluate_model(best_dt, X_train, X_test, y_train, y_test, FINAL_MODEL_NAME)


# --- Cell 14: Feature Importance Analysis ---
importances = pd.Series(best_dt.feature_importances_, index=FEATURE_COLUMNS)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

print("\nFeature Importances:\n", importances)


# --- Cell 15: FINAL MODEL SUMMARY (2-line description + metrics) ---
print("=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)
print(f"Algorithm Used : {FINAL_MODEL_NAME}")
print(f"Description    : A tree-based classifier that splits wine samples on\n"
      f"                 chemical properties (e.g. alcohol, sulphates) to\n"
      f"                 predict whether a wine is GOOD (>=7) or BAD (<7).")
print(f"Accuracy       : {final_result['accuracy']:.4f}")
print(f"Precision      : {final_result['precision']:.4f}")
print(f"Recall         : {final_result['recall']:.4f}")
print(f"F1-score       : {final_result['f1']:.4f}")
print("Confusion Matrix:")
print(final_result["confusion_matrix"])
print("=" * 60)


# --- Cell 16: PREDICT ON NEW WINE DATA (User Input) ---
def predict_new_wine(model, feature_columns):
    """
    Prompts the user to enter values for each chemical property
    of a new wine sample, then predicts GOOD or BAD quality.
    """
    print("\nEnter the following details of the new wine sample:")
    input_values = {}
    for col in feature_columns:
        while True:
            try:
                val = float(input(f"  {col}: "))
                input_values[col] = val
                break
            except ValueError:
                print("  Please enter a valid number.")

    new_sample = pd.DataFrame([input_values], columns=feature_columns)
    prediction = model.predict(new_sample)[0]
    proba = model.predict_proba(new_sample)[0]

    label = "GOOD" if prediction == 1 else "BAD"
    confidence = proba[prediction] * 100

    print("\n----- Prediction Result -----")
    print(f"Predicted Quality : {label}")
    print(f"Confidence         : {confidence:.2f}%")
    print(f"Probability (BAD)  : {proba[0]*100:.2f}%")
    print(f"Probability (GOOD) : {proba[1]*100:.2f}%")
    print("------------------------------")

    return label, proba


# Run this cell to input a new wine sample and get a prediction:
predict_new_wine(best_dt, FEATURE_COLUMNS)


# --- Cell 17 (Optional): Predict on a Hardcoded Sample Instead of Typing ---
# sample = {
#     "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
#     "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11,
#     "total sulfur dioxide": 34, "density": 0.9978, "pH": 3.51,
#     "sulphates": 0.56, "alcohol": 9.4
# }
# new_df = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
# pred = best_dt.predict(new_df)[0]
# print("GOOD" if pred == 1 else "BAD")
