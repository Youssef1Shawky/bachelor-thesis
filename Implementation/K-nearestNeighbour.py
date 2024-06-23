import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier  # Import KNN classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.pipeline import Pipeline

data = pd.read_csv('dataset.csv', encoding='latin-1')

label_encoder = LabelEncoder()
data['class_encoded'] = label_encoder.fit_transform(data['class'])

x = data['text']
y = data['class_encoded']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.fillna('')
x_test = x_test.fillna('')

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # Use KNN classifier with k=5 (you can adjust k as needed)
])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('MCC:', mcc)