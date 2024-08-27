import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Dataset
df = pd.read_csv("D:/Downloads/archive/hmeq.csv")

# Dropping rows with missing values
df = df.dropna()

# Encoding categorical variables
df = pd.get_dummies(df, drop_first=True)

# Splitting data into features X and target variable y
X = df.drop('BAD', axis=1)
y = df['BAD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applying SMOTE on training data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ---------------------Model Training-------------------- 

# Random Forest Classifier
RF = RandomForestClassifier(random_state=42)
RF.fit(X_train, y_train)


#---------------- Evaluation of Model --------------------
def plot_confusion_matrix(cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Not Bad', 'Bad'], yticklabels=['Not Bad', 'Bad'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {title}')
    plt.show()
# Random Forest Evaluation
y_pred_RF = RF.predict(X_test)
y_pred_proba_RF = RF.predict_proba(X_test)[:, 1]
accuracy_RF = accuracy_score(y_test, y_pred_RF)

print(f'Random Forest Classifier Accuracy is : {accuracy_RF}')
print('Random Forest Classifier Classification Report:')
print(classification_report(y_test, y_pred_RF))

# Confusion Matrix for Random Forest
plot_confusion_matrix(confusion_matrix(y_test, y_pred_RF), 'Random Forest')


# Feature Importance for Random Forest
importance_RF = RF.feature_importances_
indices = np.argsort(importance_RF)[::-1]
plt.figure()
plt.title("Feature Importances for Random Forest")
plt.bar(range(X.shape[1]), importance_RF[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


