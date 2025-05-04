import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Loading extracted activations and labels
X = np.load("gemma_prompt_activations.npy", allow_pickle=True)
with open("data/prompt_labels.txt", "r") as f:
    label_map = {
        "hallucination": 0,
        "deception": 1,
        "history": 2,
        "refusal": 3
    }
    y = np.array([label_map[line.strip().lower()] for line in f if line.strip()])

assert len(X) == len(y), "# of activations and labels are not corr."

# Train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LR (also ok with MLP)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Analysis：")
print(classification_report(y_test, y_pred))

joblib.dump(clf, "gemma_concept_vector_clf.pkl")
print("Saved as gemma_concept_vector_clf.pkl")
