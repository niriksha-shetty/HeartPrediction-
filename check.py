
import joblib

# Load the trained model
model = joblib.load("svm_model.joblib")

# Check the expected number of features
print("Number of expected features:", model.n_features_in_)

# Try checking if feature names are stored in the model (depends on training method)
if hasattr(model, "feature_names_in_"):
    print("Feature names used during training:")
    print(model.feature_names_in_)
else:
    print("Feature names are not stored in the model. Check your preprocessing pipeline.")


