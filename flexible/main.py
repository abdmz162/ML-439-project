import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# -------------------------
# 1. Load data
# -------------------------
dev = pd.read_csv("dev_df.csv")
test = pd.read_csv("test_df.csv")

# -------------------------
# 2. Split features/target
# -------------------------
target = "survived"

X_train = dev.drop(columns=[target])
y_train = dev[target]

X_test = test.drop(columns=[target])
y_test = test[target]   # assuming test_df.csv HAS labels

# -------------------------
# 3. Drop noisy columns
# -------------------------
drop_cols = ["name", "ticket", "cabin", "passengerid"]

X_train = X_train.drop(columns=drop_cols, errors="ignore")
X_test = X_test.drop(columns=drop_cols, errors="ignore")

# -------------------------
# 4. Feature types
# -------------------------
numeric_features = ["age", "sibsp", "parch", "fare", "pclass"]
categorical_features = ["sex", "embarked"]

numeric_features = [c for c in numeric_features if c in X_train.columns]
categorical_features = [c for c in categorical_features if c in X_train.columns]

# -------------------------
# 5. Preprocessing
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------
# 6. Model
# -------------------------
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# -------------------------
# 7. Train on FULL dev set
# -------------------------
clf.fit(X_train, y_train)

# -------------------------
# 8. Predict on test set
# -------------------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# -------------------------
# 9. FINAL EVALUATION (on test set)
# -------------------------
print("=== TEST SET PERFORMANCE ===")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_proba))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))