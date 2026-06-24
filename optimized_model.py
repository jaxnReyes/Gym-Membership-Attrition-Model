
from process_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

class OptimizedModel:
    # instantiate base model and apply grid search to find best parameters
    def __init__(self):
        self.base_model = RandomForestClassifier(
            random_state=45,
            n_jobs=-1,
        )
        self.best_model = None
        self.y_pred = None
        self.selected_features = None

    def tune_train_and_score(self, X_train, y_train, X_test, y_test):

        self.selected_features = X_train.columns

        # apply grid search parameters to find best hyperparameters
        param_grid = {
            'n_estimators': [250, 300, 350],
            'max_depth': [2, 3, 4, 5, 6, 7],
            'min_samples_split': [2, 3, 4, 5, 6, 7],
            'class_weight': ['balanced', None],
        }
        # Prioritize recall to increase sensitivity to churn - miss as few as possible
        grid = GridSearchCV(
            estimator=self.base_model,
            param_grid=param_grid,
            scoring='recall',
            n_jobs=-1,
            cv=5,
            verbose=1
        )

        grid.fit(X_train, y_train)
        self.best_model = grid.best_estimator_

        print("\n=== GRIDSEARCH OPTIMIZATION COMPLETE ===")
        print(f"Optimal Hyperparameters Discovered: {grid.best_params_}")

        self.y_pred = self.best_model.predict(X_test)

        print("\n=== DEPLOYING OPTIMIZED / TUNED MODEL ===")
        print(f"Optimized Model Accuracy: {accuracy_score(y_test, self.y_pred):.4f}\n")
        print("Optimized Classification Report:")
        print(classification_report(y_test, self.y_pred))

        return self.y_pred

    def get_optimized_importances(self):

        importances = self.best_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.selected_features,
            'Optimized_Importance': importances
        }).sort_values(by='Optimized_Importance', ascending=False)

        print("\n=== OPTIMIZED FEATURE IMPORTANCE ===")
        print(importance_df.to_string(index=False))

        return importance_df