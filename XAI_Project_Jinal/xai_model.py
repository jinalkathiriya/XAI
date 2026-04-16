import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from lime import lime_tabular

print("Starting XAI Project...")

# load dataset
df = pd.read_csv("dataset.csv")

# split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# split train test
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
random_state=42
)

# scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# accuracy
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

# save model
joblib.dump(model, "model.pkl")

# SHAP explanation
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

plt.figure()

shap.summary_plot(shap_values, X_test, show=False)

plt.savefig("shap_summary.png")

print("SHAP graph saved")

# LIME explanation
lime_explainer = lime_tabular.LimeTabularExplainer(
X_train,
feature_names=X.columns,
class_names=["No Diabetes","Diabetes"],
mode="classification"
)

exp = lime_explainer.explain_instance(
X_test[0],
model.predict_proba
)

exp.save_to_file("lime_output.html")

print("LIME explanation saved")

print("Project completed successfully")