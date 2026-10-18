import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# MLflow tracking server ka URI set karein
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris_classification_experiment")

# Data load karein
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Hyperparameters
n_estimators = 50
max_depth = 3

# MLflow run start karein
with mlflow.start_run():
    # Parameters log karein
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Model train karein
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predictions aur Evaluation
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    # Metric log karein
    mlflow.log_metric("accuracy", acc)

    # Model save / log karein
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model Training Complete! Accuracy: {acc}")