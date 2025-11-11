import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 1. load data ===
df = pd.read_csv("data/output/mushrooms_new_imputed.csv")

# === 2. preprocessing ===
X = df.drop(columns=["class"])
y = df["class"].map({"e": 0, "p": 1})   # e=edible, p=poisonous

# === 3. split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === 4. Random Forest ===
rf = RandomForestClassifier(random_state=42, n_estimators=100)
pipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ("rf", rf)
])

# === 5. Train ===
pipe.fit(X_train, y_train)

# === 6. Score ===
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === 7. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nConfusion Matrix (Formatted):")
print("                Predicted")
print("              Edible  Poisonous")
print(f"Actual Edible    {cm[0][0]:4d}      {cm[0][1]:4d}")
print(f"Actual Poisonous {cm[1][0]:4d}      {cm[1][1]:4d}")

