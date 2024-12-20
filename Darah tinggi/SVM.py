import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
data_url = r"C:\Users\jasin\OneDrive\Documents\UAS KB\Darah tinggi\heart.csv"  # Ubah path ke file dataset Anda
columns = ["age", "sex", "bmi", "systolic_bp", "diastolic_bp", "chol", "smoking", "target"]  # Sesuaikan dengan dataset
data = pd.read_csv(data_url, header=0, names=columns)  # Pastikan header sesuai

# Step 2: Split Data into Features and Labels
X = data.drop("target", axis=1)  # Fitur
y = data["target"]  # Label

# Step 3: Split Data into Training and Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train the SVM Model
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = svm_model.predict(X_test)

# Step 7: Evaluate the Model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "High BP"], yticklabels=["Normal", "High BP"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Blood Pressure Analysis")
plt.show()
