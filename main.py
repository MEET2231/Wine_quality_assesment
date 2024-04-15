import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
wine_data = pd.read_csv("winequality-red.csv")

# Drop null records
wine_data.dropna(inplace=True)

# Replace NA values in numerical columns with mean value and categorical columns with mode value as per class label
num_cols = wine_data.select_dtypes(include=np.number).columns
cat_cols = wine_data.select_dtypes(exclude=np.number).columns
for col in num_cols:
    wine_data[col] = wine_data[col].fillna(wine_data.groupby('quality')[col].transform('mean'))
for col in cat_cols:
    wine_data[col] = wine_data[col].fillna(wine_data.groupby('quality')[col].transform(lambda x: x.mode()[0]))

# Perform statistical analysis
stat_analysis = wine_data.describe()

# Display unique value counts and unique values of all columns
unique_value_counts = wine_data.nunique()
unique_values = wine_data.apply(lambda x: x.unique())

# Draw possible plots
plt.figure(figsize=(16, 12))

# Scatter plot
plt.subplot(2, 3, 1)
sns.scatterplot(x='fixed acidity', y='pH', data=wine_data)

# Line plot
plt.subplot(2, 3, 2)
sns.lineplot(x='quality', y='alcohol', data=wine_data)

# Histogram
plt.subplot(2, 3, 3)
sns.histplot(x='quality', data=wine_data)

# Box plot
plt.subplot(2, 3, 4)
sns.boxplot(x='quality', y='pH', data=wine_data)

# Violin plot
plt.subplot(2, 3, 5)
sns.violinplot(x='quality', y='alcohol', data=wine_data)

plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(wine_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Train-test split
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Conclusion
observation = "The K-nearest Neighbors Classifier achieved an accuracy of {:.2f}, precision of {:.2f}, recall of {:.2f}, and f1-measure of {:.2f}.".format(
    accuracy, precision, recall, f1)
print(observation)
