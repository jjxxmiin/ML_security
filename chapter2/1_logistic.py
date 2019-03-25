import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# read dataset
df = pd.read_csv('./datasets/payment_fraud.csv')

# one-hot encoding
df = pd.get_dummies(df, columns=['paymentMethod'])

# 데이터셋 train,test로 나누기

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label',axis=1), df['label'],
    test_size=0.33, random_state=17
)

clf = LogisticRegression().fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_test,y_pred))