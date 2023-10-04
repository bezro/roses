from pickle import load

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)

with open("./rfc.pkl", "rb") as file:
    rfc = load(file)

y_pred = rfc.predict(X)

with open("./results.csv", "w") as file:
    file.write("x,y_true,y_pred\n")
    for x, y_, y_pred_ in zip(X, y, y_pred, strict=False):
        file.write("{},{},{}\n".format(x, y_, y_pred_))

print("accuracy:", accuracy_score(y, y_pred))
print("precision:", precision_score(y, y_pred))
print("recall:", recall_score(y, y_pred))
print("roc-auc", roc_auc_score(y, y_pred))
