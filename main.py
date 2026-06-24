from baseline_model import *

# preprocess dataset
processor = ProcessData()
X_train, y_train, X_test, y_test = processor.process_data('gym_members_dataset.csv')

# evaluate baseline model
base_model = BaselineModel()
base_model.train_and_score(X_train, y_train, X_test,  y_test)
base_model.get_baseline_importance(X_test.columns)