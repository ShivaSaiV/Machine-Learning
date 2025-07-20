import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.columns = df.columns.str.lower().str.replace(" ", "_")
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ", "_")

df.churn = (df.churn == "yes").astype(int)
df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ["tenure", "monthlycharges", "totalcharges"]

full_columns = df.columns

categorical = []
for col in full_columns:
    if col not in numerical:
        if col != "churn" and col != "customerid":
            categorical.append(col)

# print(categorical)

def train(df_train, y_train, C = 1.0):
    dicts = df_train[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C = C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")

    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

C = 1.0
n_splits = 5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f" auc on fold {fold} is {auc}")
    fold = fold + 1

print("C = %s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))


# print(scores)

dv, model = train(df_full_train, df_full_train.churn.values, C = 1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(df_test.churn.values, y_pred)
print(auc)


# Save the Model

import pickle

output_file = f"model_={C}.bin"

# Open = opens a file and wb gives write perms
f_out = open(output_file, "wb")

pickle.dump((dv, model), f_out)

f_out.close()