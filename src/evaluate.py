from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)

        print(f"\n{name.upper()} RESULTS")
        print("Accuracy:", acc)
        print("ROC-AUC:", roc)
        print(classification_report(y_test, y_pred))

        results[name] = {
            "accuracy": acc,
            "roc_auc": roc
        }

    return results
