# train_model.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save to file
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
