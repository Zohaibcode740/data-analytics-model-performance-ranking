# Drug Classification – Model Performance Ranking

## 📌 Project Overview

This project focuses on building and comparing multiple Machine Learning classification models to predict the most suitable drug based on patient attributes.

The dataset contains patient information such as Age, Sex, Blood Pressure (BP), and Cholesterol level. Different classification algorithms are trained and evaluated to compare their predictive performance.

The models are ranked based on their ROC-AUC scores, helping identify the best-performing classification model.

## 🎯 Project Objectives

- Analyze and preprocess the drug classification dataset.
- Handle missing values in numerical and categorical features.
- Encode categorical variables and scale numerical data.
- Train multiple Machine Learning classification models.
- Evaluate models using performance metrics.
- Rank models according to ROC-AUC score.
- Generate a ROC curve comparison visualization.

## 📂 Dataset

The project uses the following dataset:

**File:** `DRUG - DRUG.csv`

### Features

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Sex | Gender of the patient |
| BP | Blood pressure level |
| Cholesterol | Cholesterol level |
| Drug | Target variable representing the prescribed drug |

The `Drug` column is used as the target variable, while `Age`, `Sex`, `BP`, and `Cholesterol` are used as input features.

## 🤖 Machine Learning Models

The following classification algorithms are implemented:

1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Naive Bayes
4. Random Forest
5. Decision Tree
6. Logistic Regression
7. XGBoost
8. Gradient Boosting
9. AdaBoost

Each model is trained using a preprocessing and classification pipeline.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## ⚙️ Project Workflow

1. Import required libraries.
2. Load the drug classification dataset.
3. Explore dataset shape, information, and missing values.
4. Separate input features and target variable.
5. Handle missing numerical and categorical values.
6. Encode the target variable.
7. Apply StandardScaler to numerical features.
8. Apply OneHotEncoder to categorical features.
9. Split the dataset into training and testing sets.
10. Train all classification models.
11. Generate predictions and evaluate performance.
12. Rank models based on ROC-AUC scores.
13. Save the model ranking results.
14. Plot ROC curves for model comparison.

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures how many actual positive cases are identified |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between classes |
| Confusion Matrix | Shows correct and incorrect predictions by class |

## 🏆 Model Performance Ranking

The trained models are sorted in descending order based on their ROC-AUC scores.

The final ranking includes:

- Model Rank
- Model Name
- Accuracy
- Precision
- Recall
- F1 Score
- AUC Score

The results are saved in:

`Model_Performance_Ranking.csv`

## 📈 ROC Curve Comparison

The project generates a ROC curve comparison plot for the trained models.

**Output file:** `ROC_Curve_Comparison.png`

The ROC curve visualizes the relationship between the False Positive Rate and True Positive Rate. The AUC score is used to compare the classification performance of the models.

## 📁 Project Structure

```text
data-analytics-model-performance-ranking/
│
├── DRUG - DRUG.csv
├── Model_Performance_Ranking.csv
├── ROC_Curve_Comparison.png
├── main.py
└── README.md
```

## 🚀 Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Zohaibcode740/data-analytics-model-performance-ranking.git
```

### 2. Navigate to the Project Directory

```bash
cd data-analytics-model-performance-ranking
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 4. Run the Project

```bash
python main.py
```

After execution, the project displays the model evaluation results and generates the model ranking CSV file and ROC curve visualization.

## 💡 Key Learnings

- Data preprocessing and missing-value handling
- Feature encoding and scaling
- Machine Learning model training
- Classification performance evaluation
- ROC curve and AUC score analysis
- Model comparison and ranking
- Data visualization using Matplotlib

## 👨‍💻 Author

**Zohaib**

GitHub: [@Zohaibcode740](https://github.com/Zohaibcode740)

Project Repository: [Drug Analytics – Model Performance Ranking](https://github.com/Zohaibcode740/data-analytics-model-performance-ranking)

---

⭐ If you find this project useful, feel free to star the repository.
