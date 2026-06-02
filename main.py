from src.preprocess import load_and_scale_data, split_data
from src.models import train_xgboost, build_autoencoder
from src.evaluate import plot_pr_curve
import numpy as np

# 1. Preprocess - Capture the variables here!
df = load_and_scale_data('data/creditcard.csv')
X_train, X_test, y_train, y_test = split_data(df)

# --- XGBoost Approach ---
print("Training XGBoost...")
xgb_model = train_xgboost(X_train, y_train)
xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

# --- Autoencoder Approach ---
print("Training Autoencoder on CPU...")
# 2. Now X_train is defined, so we can filter it
X_train_normal = X_train[y_train == 0]

input_dim = X_train.shape[1]
ae_model = build_autoencoder(input_dim)

# Train on normal data only
ae_model.fit(X_train_normal, X_train_normal, epochs=20, batch_size=32, verbose=0)

# 3. Get Reconstruction Error (Fraud Score)
predictions = ae_model.predict(X_test)
ae_scores = np.mean(np.power(X_test - predictions, 2), axis=1)

# --- Final Evaluation ---
print("Comparing Results...")
plot_pr_curve(y_test, xgb_scores) # Curve for XGBoost
plot_pr_curve(y_test, ae_scores)  # Curve for Autoencoder