from baseline_model import *
from optimized_model import *

# preprocess dataset
processor = ProcessData()
X_train, y_train, X_test, y_test = processor.process_data('gym_members_dataset.csv')

# evaluate baseline model
base_model = BaselineModel()
base_model.train_and_score(X_train, y_train, X_test,  y_test)
base_model.get_baseline_importance(X_test.columns)

# evaluate optimized model
optimized_model = OptimizedModel()
optimized_model.tune_train_and_score(X_train, y_train, X_test, y_test)
optimized_model.get_optimized_importances()

# display at risk members
print("\n=== GENERATING BUSINESS RETENTION LEADERBOARD ===")

test_indices = X_test.index
member_identities = processor.df_id.loc[test_indices].copy()
member_identities['Predicted_Churn_Risk'] = optimized_model.y_pred
member_identities['Actual_Churn_Status'] = y_test.values

# Filter the list down strictly to the high-risk members flagged for intervention
high_risk_list = member_identities[member_identities['Predicted_Churn_Risk'] == 1]

print(f"\nFound {len(high_risk_list)} high-risk members for immediate marketing intervention:")
print(high_risk_list[['Member_ID', 'Name', 'Actual_Churn_Status']].to_string(index=False))

