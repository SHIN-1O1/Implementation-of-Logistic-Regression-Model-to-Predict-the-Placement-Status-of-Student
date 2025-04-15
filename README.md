# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

Step 3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

Step 4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

Step 6. Stop

## Program and Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.SHERWIN INFANO 
RegisterNumber:  212224040312
*/
```
    import pandas as pd
    df=pd.read_csv('Placement_Data.csv')
    df.head()
![image](https://github.com/user-attachments/assets/d8082ba7-956e-4cb0-ac34-03e72b006399)

    d1=df.copy()
    d1=d1.drop(["sl_no","salary"],axis=1)
    d1.head()
![image](https://github.com/user-attachments/assets/cd1bba57-43a2-48f3-97e4-a0f88e1069d6)

    d1.isnull().sum()
![image](https://github.com/user-attachments/assets/64a64539-05ef-43d5-b144-ecfae1f48e04)

    d1.duplicated().sum()
![image](https://github.com/user-attachments/assets/868bc957-50f3-47b2-b167-f99c0355ef2b)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['gender']=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
```
![image](https://github.com/user-attachments/assets/32009528-944f-483e-8c12-b57e8c74dbe7)

    x=d1.iloc[:, : -1]
    x
![image](https://github.com/user-attachments/assets/4dee8975-ca47-4b06-9967-e58f802143a7)

    y=d1["status"]
    y
![image](https://github.com/user-attachments/assets/be85a4ca-8fa3-48b8-80b5-8de7b4464f0e)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/3ae4f260-3c47-46b5-9587-aa485f85f0a6)

    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
    accuracy=accuracy_score(y_test,y_pred)
    accuracy
![image](https://github.com/user-attachments/assets/ef10a2fb-63d6-415e-85c0-82cb23d02d79)

    confusion=confusion_matrix(y_test,y_pred)
    confusion
![image](https://github.com/user-attachments/assets/f2195ef4-a797-44f0-8d48-ed88542b0bef)


    from sklearn.metrics import classification_report
    classification_report=classification_report(y_test,y_pred)
    print(classification_report)
![image](https://github.com/user-attachments/assets/672865b4-a7bd-4558-8441-8cbac38cb75e)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
