
# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ==========================================
# 2. Import Dataset
# ==========================================

dataset = pd.read_csv('DRUG - DRUG.csv')

print("First 5 Rows:")
print(dataset.head())

print("\nDataset Shape:")
print(dataset.shape)

print("\nDataset Information:")
print(dataset.info())

print("\nMissing Values:")
print(dataset.isnull().sum())


# ==========================================
# 3. Data Preprocessing
# ==========================================

# Independent and Dependent Variables

X = dataset[['Age', 'Sex', 'BP', 'Cholesterol']]
y = dataset['Drug']

# Check missing values

numerical_columns = ['Age']
categorical_columns = ['Sex', 'BP', 'Cholesterol']

# Interpolation for numerical missing values

X = X.copy()

X[numerical_columns] = X[numerical_columns].interpolate(
    method='linear',
    limit_direction='both'
)

# Fill remaining missing numerical values

X[numerical_columns] = X[numerical_columns].fillna(
    X[numerical_columns].median()
)

# Fill categorical missing values

for col in categorical_columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# Encode dependent variable

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nEncoded Target Classes:")
print(label_encoder.classes_)

# Encode categorical independent variables
# Scale numerical independent variables

try:
    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse=False
    )

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', encoder, categorical_columns)
    ]
)


# ==========================================
# 4. Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1891,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Target Distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting Target Distribution:")
print(pd.Series(y_test).value_counts())


# ==========================================
# 5. Create ConfMat Function
# ==========================================

def ConfMat(y_test, y_pred, ModelName):

    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    print("\n================================")
    print("Model:", ModelName)
    print("================================")

    print("\nConfusion Matrix:")
    print(cm)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    return {
        'Model': ModelName,
        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }


# ==========================================
# 6. Model Building
# ==========================================

models = {

    'KNN': KNeighborsClassifier(n_neighbors=5),

    'SVM': SVC(probability=True, random_state=42),

    'Naive Bayes': GaussianNB(),

    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    'Decision Tree': DecisionTreeClassifier(
        random_state=42
    ),

    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        n_estimators=100,
        eval_metric='logloss',
        random_state=42
    ),

    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42
    ),

    'AdaBoost': AdaBoostClassifier(
        random_state=42
    )

}


# ==========================================
# 7. Train and Evaluate All Models
# ==========================================

results = []
trained_models = {}
roc_data = {}

for name, model in models.items():

    # Create Pipeline

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])

    # Train Model

    pipeline.fit(X_train, y_train)

    # Predictions

    y_pred = pipeline.predict(X_test)

    # Confusion Matrix Function

    result = ConfMat(y_test, y_pred, name)

    # F1 Score

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    # ROC Curve and AUC

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    result['F1 Score'] = f1
    result['AUC'] = auc

    results.append(result)

    trained_models[name] = pipeline

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    roc_data[name] = {
        'FPR': fpr,
        'TPR': tpr,
        'AUC': auc
    }


# ==========================================
# 8. Create Comprehensive DataFrame
# ==========================================

pDataFrame = pd.DataFrame(results)

# Rank models according to AUC

pDataFrame = pDataFrame.sort_values(
    by='AUC',
    ascending=False
).reset_index(drop=True)

pDataFrame.insert(
    0,
    'Rank',
    range(1, len(pDataFrame) + 1)
)

print("\n================================")
print("FINAL MODEL PERFORMANCE RANKING")
print("================================")

print(
    pDataFrame[
        [
            'Rank',
            'Model',
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score',
            'AUC'
        ]
    ].to_string(index=False)
)

# Save results

pDataFrame.to_csv(
    'Model_Performance_Ranking.csv',
    index=False
)


# ==========================================
# 9. Plot ROC Curves
# ==========================================

plt.figure(figsize=(12, 8))

for name, data in roc_data.items():

    plt.plot(
        data['FPR'],
        data['TPR'],
        label=f"{name} (AUC = {data['AUC']:.3f})"
    )

# Random Classifier Line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='Random Classifier'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve Comparison of All ML Models')

plt.legend(loc='lower right')

plt.grid(alpha=0.3)

plt.savefig('ROC_Curve_Comparison.png')

plt.show()


# ==========================================
# 10. Best Model
# ==========================================

best_model = pDataFrame.iloc[0]

print("\n================================")
print("BEST PERFORMING MODEL")
print("================================")

print("Model Name:", best_model['Model'])
print("Accuracy:", best_model['Accuracy'])
print("Precision:", best_model['Precision'])
print("Recall:", best_model['Recall'])
print("F1 Score:", best_model['F1 Score'])
print("AUC Score:", best_model['AUC'])

