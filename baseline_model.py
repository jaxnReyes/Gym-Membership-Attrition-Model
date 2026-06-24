from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from process_data import *

class BaselineModel:
    # instantiate baseline model using all cpu cores available
    def __init__(self):
        self.model = RandomForestClassifier(
            random_state=45,
            n_jobs=-1
        )

    def train_and_score(self, X_train, y_train, X_test, y_test):
        # fit model with preprocessed training data
        self.model.fit(X_train, y_train)

        # predict on test split
        y_pred = self.model.predict(X_test)

        print("=== DEPLOYING BASELINE MODEL ===")
        print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        print("Baseline Classification Report:")
        print(classification_report(y_test, y_pred))

        return y_pred

    def get_baseline_importance(self, feature_names):
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Baseline_Importance': importances
        }).sort_values(by='Baseline_Importance', ascending=False)

        print("=== BASELINE FEATURE IMPORTANCE ===")
        print(importance_df.to_string(index=False))
        print("\n")

        return importance_df