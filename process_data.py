import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

class ProcessData:

    def __init__(self):
        self.df_id = None

    def process_data(self, dataset_path):
        df = pd.read_csv(dataset_path)

        # copy member identification for final results
        self.df_id = df[['Member_ID', 'Name']].copy()

        # map favorite exercise to either resistance or cardio
        exercise_map = {
            'Pull-ups': 'Resistance',
            'Squats': 'Resistance',
            'Bench Press': 'Resistance',
            'Deadlift': 'Resistance',
            'Treadmill': 'Cardio',
            'Cycling': 'Cardio',
        }

        # add favorite exercise type to dataframe
        df['Exercise_Type'] = df['Favorite_Exercise'].map(exercise_map)
        # remove irrelevant feature columns
        X = df.drop(columns=['Member_ID', 'Name', 'Address', 'Phone_Number', 'Favorite_Exercise'])

        # map churn target variable to binary classification
        Y = df['Churn'].map({'Yes': 1, 'No': 0})

        # convert date info to datetime
        date_cols = ['Join_Date', 'Last_Visit_Date']
        for col in date_cols:
            X[col] = pd.to_datetime(X[col])

        # calculate length of membership
        # for live production, we would set this relative to the current date
        X['Membership_Length'] = (X['Last_Visit_Date'] - X['Join_Date']).dt.days
        # we would also use last visit date to determine how many days since the last visit relative to today
        # days since last visit cannot be determined since we have no cancellation date for churned members

        # remove date information after finding membership length
        X = X.drop(columns=date_cols + ['Churn'])

        # define column types for processing split
        num_cols = ['Age', 'Avg_Workout_Duration_Min', 'Avg_Calories_Burned',
                    'Total_Weight_Lifted_kg', 'Visits_Per_Month', 'Membership_Length']
        cat_cols = ['Gender', 'Membership_Type', 'Exercise_Type']

        # split column types into separate transformers
        # numbers columns with missing data will be imputed with median values to minimize outliers
        #  category columns will be transformed with one hot encoding
        preprocessor = ColumnTransformer(transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ])
        X_processed = preprocessor.fit_transform(X)

        # reconstruct dataframe to include column names after transformation
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        all_features = np.concatenate([num_cols, cat_features])
        X_clean = pd.DataFrame(X_processed, columns=all_features)

        # split data into training and test (80/20)
        # stratify to preserve churn ratio on data splits
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_clean,
            Y,
            test_size=0.2,
            stratify=Y,
            random_state=125,
        )

        from collections import Counter
        print("=== PRIOR TO SMOTE BALANCING ===")
        print(f"Training target class distribution: {Counter(Y_train)}")

        # implement SMOTE to balance churn ratio on training data only
        smote = SMOTE(random_state=45)
        X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)

        print("\n=== AFTER SMOTE BALANCING ===")
        print(f"Balanced Training target class distribution: {Counter(Y_train_balanced)}")
        print(f"Final training features matrix shape: {X_train_balanced.shape}\n")

        return X_train_balanced, Y_train_balanced, X_test, Y_test