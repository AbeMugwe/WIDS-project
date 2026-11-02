import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

FEATURES = [
    "frames_per_second",
    "deauth_count",
    "management_ratio",
]

# Training data: Deauth_29
train = pd.read_csv("features.csv")

# Cross-chunk test data: Deauth_30
test = pd.read_csv("test_features_30.csv")

X_train = train[FEATURES]
y_train = train["label"]

X_test = test[FEATURES]
y_test = test["label"]

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    ),
}

best_model = None
best_name = None
best_recall = -1

for name, model in models.items():

    # 1. Train the model
    model.fit(X_train, y_train)

    # 2. Make predictions
    predictions = model.predict(X_test)

    # 3. Evaluate the model
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        output_dict=True,
        zero_division=0
    )

    attack_recall = report["Attack"]["recall"]

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Attack recall: {attack_recall:.4f}")

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ))

    print("Feature importance:")
    for feature, importance in zip(
        FEATURES, model.feature_importances_
    ):
        print(f"  {feature}: {importance:.4f}")

    # 4. Select the model with the highest attack recall
    if attack_recall > best_recall:
        best_recall = attack_recall
        best_model = model
        best_name = name


# 5. Save the best model AFTER evaluating both
joblib.dump({
    "model": best_model,
    "features": FEATURES,
    "model_name": best_name,
    "training_source": "Deauth_29.csv"
}, "model_v2.joblib")

print("\n" + "=" * 45)
print("BEST MODEL SELECTED:", best_name)
print(f"Attack recall: {best_recall:.4f}")
print("Saved as model_v2.joblib")
