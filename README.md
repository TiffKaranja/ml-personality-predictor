# ML Personality Predictor

This project uses machine learning to classify individuals as **Introverts** or **Extroverts** based on features like time spent alone, stage fear, social event attendance, and more. The data is processed, modeled, and evaluated using several classification algorithms, with **SVM (RBF Kernel)** emerging as the best performer.

## 📊 Dataset Overview

The dataset contains 750 observations and features such as:

- Time_spent_Alone
- Stage_fear
- Social_event_attendance
- Going_outside
- Drained_after_socializing
- Friends_circle_size
- Post_frequency

Target variable: `Personality_Type` (Introvert or Extrovert)

## 🧪 Process Summary

### Section 1–5: Preprocessing & Baseline Model
- Handled categorical and numerical data
- Scaled features using `StandardScaler`
- Trained an initial **Logistic Regression** model
- Evaluated using accuracy, confusion matrix, classification report

### Section 6: Alternative Classifiers
- **Decision Tree** (accuracy: 85%)
- **K-Nearest Neighbors (k=5)** (accuracy: 92%)
- **SVM (RBF Kernel)** (accuracy: **93%**)
- Compared models using:
  - Confusion matrices
  - ROC curves
  - AUC scores (SVM had the highest: 0.952)

### Section 7: Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation
- Tuned:
  - `max_depth` and `min_samples_split` for Decision Tree
  - `C` and `gamma` for SVM
- Best estimator: **Tuned SVM** with improved performance

## 🧠 Final Model Performance (SVM with RBF Kernel)
- **Accuracy:** 93%
- **Precision (Extrovert):** 0.93
- **Recall (Introvert):** 0.91
- **AUC Score:** 0.952

## 📈 Key Visualizations
- Confusion matrices
- ROC curves for all models
- Validation curves to show tuning effects

## 🚀 Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (StandardScaler, GridSearchCV, Classification Models)
