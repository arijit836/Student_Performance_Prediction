# 🎓 Early Warning System for Student Academic Performance Using Machine Learning

<p align="center">
  <b>A Machine Learning-Based Student Academic Risk Prediction System</b>
</p>

---

## 📌 1. Project Overview

The Early Warning System for Student Academic Performance is a Machine Learning-based project designed to identify students who may be at academic risk.

The system analyzes student-related academic and personal information and predicts whether a student belongs to the **At Risk** or **Not At Risk** category.

It aims to support early identification and encourage timely academic assistance.

## 🎯 2. Project Objectives

- Identify students who may be academically at risk.
- Apply Machine Learning classification techniques.
- Compare multiple classification algorithms.
- Evaluate model performance using standard metrics.
- Develop a user-friendly web application for prediction.

## 📊 3. Dataset Description

**Dataset:** UCI Student Performance Dataset (Mathematics)

- Total Records: 395
- Total Input Features: 30
- Target Variable: Risk

### 🎯 Target Classification

- **At Risk (1):** Final grade (G3) below 10.
- **Not At Risk (0):** Final grade (G3) equal to or above 10.

The G1, G2, and G3 columns are excluded from model inputs to prevent target leakage.

## 🛠️ 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Web Application |
| Joblib | Model Saving and Loading |

## 🤖 5. Machine Learning Algorithms

Three classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models are trained using an 80:20 train-test split.

## 📈 6. Model Evaluation

The following performance metrics are used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 🏆 Model Performance Comparison

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 69.62% | 60.00% |
| Random Forest | 70.89% | 51.06% |
| Decision Tree | 56.96% | 32.00% |

**Selected Model:** Logistic Regression

It is selected based on the highest F1 Score among the three evaluated models.

## 🌐 7. Web Application

A user-friendly Streamlit web application is developed for student academic risk prediction.

### ✨ Key Features

- Student information input form.
- Machine Learning-based prediction.
- At Risk and Not At Risk results.
- Simple and interactive interface.

## 📁 8. Project Structure

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
│   │   ├── confusion_matrices.png
│   │   └── EDA graphs
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
└── README.md
```

## ⚠️ 9. Limitations

- The dataset contains information about Portuguese school students.
- The model is developed for educational and experimental purposes.
- Predictions are not guaranteed to be accurate for every student.
- Further validation using local institutional data is necessary.

## 🔮 10. Future Improvements

- Use larger and more diverse student datasets.
- Improve model performance through hyperparameter tuning.
- Add more data visualization features.
- Introduce additional Machine Learning algorithms.
- Improve the web application's user interface.

## ✅ 11. Conclusion

This project demonstrates the practical application of Machine Learning in educational data analysis.

By comparing three classification algorithms, the system selects Logistic Regression based on its F1 Score and provides an interactive web application for predicting student academic risk.

The proposed system can serve as an experimental tool to support early identification of students who may need additional academic assistance.

---

<p align="center">
  🎓 <b>Machine Learning Mini Project</b>
  <br>
  <i>Early Identification • Smart Prediction • Academic Support</i>
</p>