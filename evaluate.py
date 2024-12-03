import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test_seq, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    y_pred_probs = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{output_dir}/confusion_matrix.png")

    class_report = classification_report(y_true, y_pred, output_dict=True)
    with open(f"{output_dir}/classification_report.json", "w") as f:
        json.dump(class_report, f)
