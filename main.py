from process_data import *

processor = ProcessData()
X_train, X_test, y_train, y_test = processor.process_data('gym_members_dataset.csv')