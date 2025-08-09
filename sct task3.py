# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree

# Load dataset
df = pd.read_csv("/Users/TEC/Desktop/New folder (4)/bank_marketing_sample.csv")

# Step 1: Show first few rows
print("🟩 Dataset Head:")
print(df.head())

# Step 2: Check for missing values
print("\n🟩 Missing Values:")
print(df.isnull().sum())

# Step 3: Encode categorical features
label_encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# Step 4: Define features and target
X = df.drop('y', axis=1)
y = df['y']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Decision Tree Classifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)

print("\n🟩 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🟩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🟩 Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Plot the Decision Tree
plt.figure(figsize=(15, 8))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title("Decision Tree - Bank Marketing")
plt.show()
