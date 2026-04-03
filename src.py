import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('healthcare_dataset.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

sns.countplot(x='diabetes', data=df)
plt.show()

sns.histplot(df['bmi'], kde=True)
plt.show()

sns.boxplot(x='diabetes', y='age', data=df)
plt.show()

sns.scatterplot(x='exercise_hours_per_week', y='bmi', hue='diabetes', data=df)
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

X = df.drop(['diabetes', 'patient_id'], axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

nn_model = Sequential()

nn_model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

loss, accuracy = nn_model.evaluate(X_test, y_test)


print("Neural Network Accuracy:", accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.show()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
