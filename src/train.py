from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def train_models(X_train, y_train):
    models = {}

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    models["xgboost"] = xgb

    return models
