# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect the file encoding using the chardet library  
2. Load the dataset using the detected encoding  
3. Extract features (messages) and labels (spam/ham) from the dataset  
4. Convert text data into numeric form using CountVectorizer and split into train/test sets  
5. Train the SVM model, make predictions, and evaluate the results

## Program:

Program to implement the SVM For Spam Mail Detection.

Developed by: A Ahil Santo

RegisterNumber: 212224040018

```
import chardet
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

file = r"E:\Desktop\CSE\Introduction To Machine Learning\dataset\spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

data = pd.read_csv(file, encoding='Windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

x = data["v1"].values
y = data["v2"].values

cv = CountVectorizer()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("Predicted Labels:\n", y_pred)

```

## Output:

Result

![1](https://github.com/user-attachments/assets/6b122e33-623b-4e9f-9986-c1a42e87be04)

Head

![2](https://github.com/user-attachments/assets/b50beb0c-e415-42c3-a516-d06123030260)

Info

![3](https://github.com/user-attachments/assets/db99ed98-e858-46ee-9a3d-abf3d3fd8457)

Null Values

![4](https://github.com/user-attachments/assets/d1e3f7b9-c418-45bf-bc7d-9f6e7fff5d41)

Y_Prediction

![5](https://github.com/user-attachments/assets/deb06042-8adf-456d-b010-4af56de30437)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
