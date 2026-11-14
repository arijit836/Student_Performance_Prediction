# 🎓 Early Warning System for Student Academic Performance Using Machine Learning

## 📌 1. Project Overview

The **Early Warning System for Student Academic Performance** is a Machine Learning-based project designed to identify students who may be academically at risk.

The system analyzes student-related information and classifies students into two categories:

- 🔴 **At Risk (1):** Final Mathematics grade (G3) is below 10.
- 🟢 **Not At Risk (0):** Final Mathematics grade (G3) is 10 or above.

The main purpose is to provide an early indication of possible academic difficulties and help identify students who may need additional academic support.

---

## 🎯 2. Problem Statement

Students who are facing academic difficulties may not always be identified at an early stage. Without early identification, it can be difficult to provide timely academic support.

To address this problem, this project develops a Machine Learning-based prediction system that analyzes student information and classifies students as At Risk or Not At Risk.

### 🔍 Problems Addressed

- Identify students who may be academically at risk.
- Analyze student-related information.
- Compare different Machine Learning algorithms.
- Provide a simple prediction interface.
- Support early academic attention and assistance.

The system provides an early indication, not a final judgement about a student's academic ability or future performance.

---

## 🎯 3. Project Objectives

- Inspect and understand the student performance dataset.
- Perform data preprocessing and cleaning.
- Analyze student information using Exploratory Data Analysis (EDA).
- Implement and compare three Machine Learning algorithms.
- Evaluate model performance using standard metrics.
- Select and save the best-performing model.
- Develop a user-friendly Streamlit web application.

---

## 📊 4. Dataset Description

**Dataset:** UCI Student Performance Dataset (Mathematics)

The original dataset contains:

- 📌 Total Records: 395
- 📌 Total Columns: 33
- 📌 Input Features: 30
- 📌 Target Variable: Risk

The dataset contains student-related information, including personal background, family information, study habits, absences, and academic performance.

### 🎯 Target Classification

The Risk column is created using the final grade (G3).

| Risk Value | Category | Condition |
|:---:|:---:|:---:|
| 1 | 🔴 At Risk | G3 below 10 |
| 0 | 🟢 Not At Risk | G3 equal to or above 10 |

The G1, G2, and G3 columns are excluded from model input features to prevent target leakage.

After preprocessing, the dataset contains 395 records, 30 input features, and one target column.

---

## 🛠️ 5. Technologies Used

| Technology | Purpose |
|:---|:---|
| 🐍 Python | Programming Language |
| 🐼 Pandas | Data Processing |
| 🔢 NumPy | Numerical Operations |
| 📈 Matplotlib | Data Visualization |
| 📊 Seaborn | Statistical Visualization |
| 🤖 Scikit-learn | Machine Learning |
| 🌐 Streamlit | Web Application |
| 💾 Joblib | Model Saving and Loading |

---

## 🔄 6. Project Workflow

The project follows these main steps:

1. 🔍 **Data Inspection:** Understanding the dataset, checking data types, missing values, and duplicate records.
2. 🧹 **Data Preprocessing:** Preparing the dataset and creating the Risk target variable.
3. 📊 **Exploratory Data Analysis:** Analyzing student information through different graphs.
4. 🤖 **Model Training:** Training three Machine Learning classification algorithms.
5. 📈 **Model Evaluation:** Comparing model performance using standard evaluation metrics.
6. 🧪 **Model Testing:** Loading the saved model and checking its prediction functionality.
7. 🌐 **Web Application:** Developing an interactive application for student risk prediction.

---

## 🤖 7. Machine Learning Algorithms

Three classification algorithms are implemented in this project:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models use an 80:20 train-test split with stratified sampling.

The preprocessing pipeline uses StandardScaler for numerical features and OneHotEncoder for categorical features.

---

## 📈 8. Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 🏆 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|:---|---:|---:|---:|---:|
| Logistic Regression | 69.62% | 52.94% | 69.23% | 60.00% |
| Decision Tree | 56.96% | 33.33% | 30.77% | 32.00% |
| Random Forest | 70.89% | 57.14% | 46.15% | 51.06% |

### ⭐ Selected Model

**Logistic Regression** is selected as the final model because it achieved the highest F1-score of 60.00% among the three evaluated models.

Random Forest achieved the highest accuracy of 70.89%.

The final model is selected based on F1-score because it balances Precision and Recall.

---

## 📊 9. Output Graphs

The project generates different graphs to understand the dataset and evaluate model performance.

The generated graphs include:

- 📊 **risk_distribution.png:** Shows the distribution of At Risk and Not At Risk students.
- 📈 **age_distribution.png:** Shows the age distribution of students.
- 📉 **absences_vs_risk.png:** Shows the relationship between student absences and academic risk.
- 📉 **failures_vs_risk.png:** Shows the relationship between previous failures and academic risk.
- 📚 **studytime_vs_risk.png:** Shows the relationship between study time and academic risk.
- 🎨 **correlation_heatmap.png:** Shows the correlation between numerical features.
- 🎯 **confusion_matrices.png:** Shows the prediction performance of the evaluated models.

The model comparison results are also saved in `model_comparison.csv`.

---

## 🌐 10. Web Application

A user-friendly web application is developed using Streamlit.

### ✨ Key Features

- 📝 Student information input form.
- 🤖 Machine Learning-based risk prediction.
- 🔴 At Risk and 🟢 Not At Risk prediction results.
- 💻 Simple and interactive user interface.

The application allows users to enter student information and receive a predicted academic risk category.

---

## 📁 11. Project Structure

```text
Student_Performance_Prediction/
│
├── data/
│   ├── raw/
│   │   └── student-mat.csv
│   └── processed/
│       └── student-performance-processed.csv
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   ├── figures/
│   │   ├── absences_vs_risk.png
│   │   ├── age_distribution.png
│   │   ├── confusion_matrices.png
│   │   ├── correlation_heatmap.png
│   │   ├── failures_vs_risk.png
│   │   ├── risk_distribution.png
│   │   └── studytime_vs_risk.png
│   │
│   └── model_comparison.csv
│
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── eda.py
│   ├── train.py
│   └── test_model.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ 12. How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Data Inspection

```bash
python src/inspect_data.py
```

### Step 3: Run Data Preprocessing

```bash
python src/prepare_data.py
```

### Step 4: Run Exploratory Data Analysis

```bash
python src/eda.py
```

### Step 5: Train the Models

```bash
python src/train.py
```

### Step 6: Test the Saved Model

```bash
python src/test_model.py
```

### Step 7: Start the Web Application

```bash
streamlit run app.py
```

Make sure the dataset is available in the correct folder before running the project.

---

## ⚠️ 13. Project Limitations

- The dataset contains only 395 student records from Portuguese secondary schools.
- The dataset may not represent students from other countries or educational systems.
- The Risk classification is based on a fixed grade threshold of 10.
- The model uses only the available dataset features.
- The model's performance may change when tested with new datasets.
- Predictions are not guaranteed to be correct for every student.
- The system is an experimental educational support tool, not a replacement for teachers' judgement.
- Further testing with larger and more diverse datasets is required before real-world use.

---

## 🚀 14. Future Improvements

- Use larger and more diverse student datasets.
- Test the model using local institutional data.
- Improve model performance through hyperparameter tuning.
- Explore additional Machine Learning algorithms.
- Add more data visualization features.
- Improve the web application's user interface.
- Perform further validation using real-world student data.

---

## 🏁 15. Conclusion

This project demonstrates the practical application of Machine Learning for identifying possible academic risk among students.

It includes data inspection, preprocessing, Exploratory Data Analysis, model training, performance comparison, model testing, and a Streamlit web application.

Among the three evaluated algorithms, Logistic Regression achieved the highest F1-score and was selected as the final model.

The proposed system can provide an early indication of academic risk and support timely academic assistance.

---

## 📚 16. Dataset Reference

The project uses the UCI Student Performance Dataset.

**Reference:**

Cortez, P. and Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of the 5th Future Business Technology Conference.

**Dataset Source:** UCI Machine Learning Repository – Student Performance Dataset.

---

<div align="center">

## 👨‍💻 Project Information

**Project Title:** Early Warning System for Student Academic Performance Using Machine Learning

**Developer:** Arijit Maity

🎓 Early Identification • 🤖 Smart Prediction • 📚 Academic Support

</div>
