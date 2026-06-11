import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.base import clone

baseline = pd.read_csv("spring2026_kaggle_linear_regression_challenge_train.csv")
prod = pd.read_csv("spring2026_kaggle_linear_regression_challenge_train.csv")

features = [f'x{i}' for i in range(15)]
X = baseline[features].copy()
X['x9_sq'] = X['x9'] ** 2
X['x9_abs'] = X['x9'].abs()
y = baseline['target'].values

q1, q3 = np.percentile(y, [25, 75])
iqr = q3 - q1
mask_clean = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Model A: SVR trained on clean data
pipe_a = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', SVR(kernel='rbf', C=1000, epsilon=3.0, gamma=0.001))
])

# Model B: OLS on x9^3
x9_cubed_baseline = baseline['x9'].fillna(0).values.reshape(-1, 1) ** 3

# Cross-validate different blending weights
print("=== Ensemble: Clean SVR + x9^3 OLS with blending weights ===\n")

best_r2 = -999
best_weight = None
best_preds = None

for weight_b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    results = np.zeros(len(y))
    for train_idx, test_idx in kf.split(X.values):
        # Model A: train on clean subset
        train_clean = train_idx[mask_clean[train_idx]]
        model_a = clone(pipe_a)
        model_a.fit(X.values[train_clean], y[train_clean])
        pred_a = model_a.predict(X.values[test_idx])

        # Model B: OLS x9^3 on all training data
        model_b = LinearRegression()
        model_b.fit(x9_cubed_baseline[train_idx], y[train_idx])
        pred_b = model_b.predict(x9_cubed_baseline[test_idx])

        # Blend
        results[test_idx] = (1 - weight_b) * pred_a + weight_b * pred_b

    r2 = r2_score(y, results)
    r2_c = r2_score(y[mask_clean], results[mask_clean])
    r2_o = r2_score(y[~mask_clean], results[~mask_clean])
    print(f"  weight_B={weight_b:.1f}: R2={r2:.4f} (clean={r2_c:.4f}, outlier={r2_o:.4f})")
    if r2 > best_r2:
        best_r2 = r2
        best_weight = weight_b
        best_preds = results.copy()

print(f"\n  Best blend weight: {best_weight}, R2={best_r2:.4f}")

# Also try with prediction clipping
print("\n=== Ensemble + prediction clipping ===\n")
best_clip_r2 = -999
best_clip_pct = None
best_clip_weight = None

for weight_b in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for clip_pct in [95, 97, 99, 99.5]:
        results = np.zeros(len(y))
        for train_idx, test_idx in kf.split(X.values):
            train_clean = train_idx[mask_clean[train_idx]]
            model_a = clone(pipe_a)
            model_a.fit(X.values[train_clean], y[train_clean])
            pred_a = model_a.predict(X.values[test_idx])

            model_b = LinearRegression()
            model_b.fit(x9_cubed_baseline[train_idx], y[train_idx])
            pred_b = model_b.predict(x9_cubed_baseline[test_idx])

            blended = (1 - weight_b) * pred_a + weight_b * pred_b

            # Clip predictions
            low = np.percentile(y[train_idx], (100 - clip_pct) / 2)
            high = np.percentile(y[train_idx], 100 - (100 - clip_pct) / 2)
            results[test_idx] = np.clip(blended, low, high)

        r2 = r2_score(y, results)
        if r2 > best_clip_r2:
            best_clip_r2 = r2
            best_clip_pct = clip_pct
            best_clip_weight = weight_b

print(f"  Best clipped: weight_B={best_clip_weight}, clip_pct={best_clip_pct}, R2={best_clip_r2:.4f}")

# Pick final approach
if best_clip_r2 > best_r2:
    final_r2 = best_clip_r2
    use_clip = True
    final_weight = best_clip_weight
    final_clip_pct = best_clip_pct
    print(f"\n  Using clipped ensemble (R2={final_r2:.4f})")
else:
    final_r2 = best_r2
    use_clip = False
    final_weight = best_weight
    final_clip_pct = None
    print(f"\n  Using unclipped ensemble (R2={final_r2:.4f})")

# --- Train final model and predict production ---
print("\n" + "=" * 60)
print("TRAINING FINAL MODEL AND PREDICTING PRODUCTION")
print("=" * 60)

# Model A: train on all clean baseline data
pipe_a_final = clone(pipe_a)
pipe_a_final.fit(X[mask_clean].values, y[mask_clean])
pred_a_prod = pipe_a_final.predict(
    prod[features].assign(x9_sq=prod['x9'] ** 2, x9_abs=prod['x9'].abs()).values
)

# Model B: train on all baseline data
x9_cubed_prod = prod['x9'].fillna(0).values.reshape(-1, 1) ** 3
model_b_final = LinearRegression()
model_b_final.fit(x9_cubed_baseline, y)
pred_b_prod = model_b_final.predict(x9_cubed_prod)

# Blend
y_pred_prod = (1 - final_weight) * pred_a_prod + final_weight * pred_b_prod

# Clip if needed
if use_clip:
    low = np.percentile(y, (100 - final_clip_pct) / 2)
    high = np.percentile(y, 100 - (100 - final_clip_pct) / 2)
    y_pred_prod = np.clip(y_pred_prod, low, high)

print(f"\nPrediction stats:")
print(f"  Mean: {y_pred_prod.mean():.2f}")
print(f"  Std:  {y_pred_prod.std():.2f}")
print(f"  Min:  {y_pred_prod.min():.2f}")
print(f"  Max:  {y_pred_prod.max():.2f}")

# Save output
model_name = "ensemble"
output = pd.DataFrame({'Id': prod['Id'], 'target': y_pred_prod})
output.to_csv(f"output_{model_name}.csv", index=False)
print(f"\nSaved predictions to output_{model_name}.csv ({len(output)} rows)")

# Save params
params_dict = {
    "model": "Ensemble (Clean SVR + x9^3 OLS)",
    "blend_weight_B": final_weight,
    "clip_percentile": final_clip_pct,
    "model_a": {
        "type": "SVR",
        "kernel": "rbf",
        "C": 1000,
        "epsilon": 3.0,
        "gamma": 0.001,
        "trained_on": "clean subset (IQR outlier removal)"
    },
    "model_b": {
        "type": "LinearRegression",
        "features": ["x9^3"],
        "trained_on": "all data"
    },
    "cv_r2": round(final_r2, 4)
}

with open(f"params_{model_name}.json", "w") as f:
    json.dump(params_dict, f, indent=2)
print(f"Saved hyperparameters to params_{model_name}.json")
print(json.dumps(params_dict, indent=2))
