<strong> **DO NOT DISTRIBUTE OR PUBLICLY POST SOLUTIONS TO THESE LABS. MAKE ALL FORKS OF THIS REPOSITORY WITH SOLUTION CODE PRIVATE. PLEASE REFER TO THE STUDENT CODE OF CONDUCT AND ETHICAL EXPECTATIONS FOR COLLEGE OF INFORMATION TECHNOLOGY STUDENTS FOR SPECIFICS. ** </strong>

# WESTERN GOVERNORS UNIVERSITY

## D683 – ADVANCED AI AND ML

Welcome to Advanced AI and ML!

For specific task instructions and requirements for this assessment, please refer to the course page.

######

# Gym Membership Attrition Prediction

This advanced machine learning pipeline is designed to predict gym membership churn risk using behavioral, physical, and contract-based member data. This project is built using a regularized Random Forest Binary Classifier and optimized for high-sensitivity recall to minimize non-detection of at-risk members.

---

## System Requirements & Dependencies

This project is built on Python 3.12 and a core suite of scientific computing and machine learning libraries.

### Core Libraries Used:
* **Pandas & NumPy:** For data engineering, datetime processing, and vector transformations.
* **Scikit-Learn:** For machine learning workflows (Imputation, One-Hot Encoding, GridSearch Cross-Validation, RFECV).
* **Imbalanced-Learn (SMOTE):** For resolving severe minority class imbalance prior to model evaluation.

---

## Step-by-Step Installation & Setup

Follow these operational steps to provision your local environment and replicate the validated project results.

### 1. Extract the Project Files
Ensure your directory structure contains the following core files in the root folder:
```
membership_attrition/ 
|
├── gym_members_dataset.csv     
├── process_data.py             
├── baseline_model.py          
├── optimized_model.py        
├── main.py                    
├── requirements.txt           
└── README.md         
```
### 2. Create and Activate a Virtual Environment

### 3. Install Dependencies Listed in Requirements.txt    

### 4. Run Program Through Preferred IDE or Within Terminal 
