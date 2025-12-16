# --------------------------
# Import bibliotek
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------
# Wczytanie danych
# --------------------------
df = pd.read_csv("weather.csv")
print("Rozmiar danych:", df.shape)
df.head()

# --------------------------
# Podstawowa eksploracja danych
# --------------------------
print(df.info())
print(df.describe())
print(df.isnull().sum().sort_values(ascending=False))
print(df['RainTomorrow'].value_counts())

# --------------------------
# Wizualizacja braków danych
# --------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Mapa braków danych")
plt.show()

# --------------------------
# Analiza korelacji zmiennych numerycznych
# --------------------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(12, 10))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Macierz korelacji zmiennych numerycznych")
plt.show()

# --------------------------
# Usunięcie nieistotnych kolumn
# --------------------------
df = df.drop(columns=["Date", "Location", "RainToday", "RISK_MM"])

# --------------------------
# Konwersja zmiennej celu
# --------------------------
df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})

# --------------------------
# Identyfikacja kolumn numerycznych i kategorycznych
# --------------------------
num_cols = df.select_dtypes(include=np.number).columns.drop("RainTomorrow")
cat_cols = df.select_dtypes(include="object").columns
print("Cechy numeryczne:", num_cols)
print("Cechy kategoryczne:", cat_cols)

# --------------------------
# Podział danych na X i y
# --------------------------
X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

# --------------------------
# Podział na zbiór treningowy i testowy
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Pipeline dla cech numerycznych i kategorycznych
# --------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# --------------------------
# Definicja modeli
# --------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

# --------------------------
# Trenowanie modeli, predykcje i cross-validation
# --------------------------
results = []
results_cv = []

for name, clf in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Fit na zbiorze treningowym
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Ewaluacja na zbiorze testowym
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, prec, rec, f1])

    print(f"\n{name} - Test set")
    print("-" * 40)
    print(classification_report(y_test, y_pred))

    # Macierz konfuzji
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Cross-validation (F1-score)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    results_cv.append([name, mean_cv, std_cv])
    print(f"{name} - Cross-validated F1-score: {mean_cv:.3f} ± {std_cv:.3f}")

# --------------------------
# Porównanie modeli w tabelach
# --------------------------
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
results_df = results_df.sort_values(by="F1-score", ascending=False)
print("\nPorównanie modeli - Test set:")
print(results_df)

results_cv_df = pd.DataFrame(results_cv, columns=["Model", "F1-score (mean)", "F1-score (std)"])
print("\nPorównanie modeli - Walidacja krzyżowa:")
print(results_cv_df)
