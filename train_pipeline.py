
# **Author**: Saurabh maurya
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# %% [code] -- Data Loading & Validation
# %% [code] -- Data Loading & Validation
def load_data(path):
    """Optimized data loader with dtype conversion"""
    df = pd.read_parquet(path)
    for col in df.select_dtypes(include='float64'):
        df[col] = df[col].astype('float32')
    return df

# ===============================
# 🔽 UPDATE THESE PATHS LOCALLY
# ===============================
TRAIN_PATH = "data/train.parquet"
TEST_PATH = "data/test.parquet"
SAMPLE_SUB_PATH = "data/sample_submission.parquet"

try:
    train = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)
    sample_sub = load_data(SAMPLE_SUB_PATH)
except Exception as e:
    raise SystemExit(f"Data loading failed: {str(e)}")
  z
# Convert and validate target column
train['target'] = train['target'].astype('int8')
if train['target'].nunique() != 2:
    raise ValueError("Invalid target values - must contain exactly 2 classes")

print("Dataset Shapes:")
print(f"Train: {train.shape}, Test: {test.shape}")
print("\nClass Distribution:")
print(train['target'].value_counts(normalize=True))

# %% [code] -- Feature Engineering & Preprocessing
def process_datetime(df):
    """Optimized datetime feature engineering"""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['hour'] = df['Date'].dt.hour.astype('int8')
    df['day_of_week'] = df['Date'].dt.dayofweek.astype('int8')
    df['month'] = df['Date'].dt.month.astype('int8')
    
    # Add interaction feature
    df['X1_X2_ratio'] = df['X1'] / (df['X2'] + 1e-6)
    return df.drop('Date', axis=1)

# Process data
train = process_datetime(train)
test = process_datetime(test)

# Handle outliers
sensor_cols = [f'X{i}' for i in range(1,6)]
for col in sensor_cols:
    q1, q3 = np.percentile(train[col], [5, 95])
    iqr = q3 - q1
    train[col] = np.clip(train[col], q1 - 1.5*iqr, q3 + 1.5*iqr)
    test[col] = np.clip(test[col], q1 - 1.5*iqr, q3 + 1.5*iqr)

# %% [code] -- Model Development
# Prepare data
X = train.drop('target', axis=1)
y = train['target']
test_final = test.drop('ID', axis=1)

# Initialize models with MASSIVELY increased learning and training parameters
models = {
    'XGBoost': XGBClassifier(
        scale_pos_weight=len(y[y==0])/len(y[y==1]),
        tree_method='gpu_hist',
        eval_metric='logloss',
        use_label_encoder=False,
        learning_rate=0.005,  # Further reduced for even slower learning
        n_estimators=2000,    # Massive increase to allow for slow learning rate
        subsample=0.8,
        colsample_bytree=0.8
    ),
    'RandomForest': RandomForestClassifier(
        class_weight='balanced',
        n_jobs=-1,
        max_samples=0.3,
        n_estimators=1000     # Doubled from previous increase
    )
}

# Training pipeline with even more intensive cross-validation
tscv = TimeSeriesSplit(n_splits=15)  # Further increased from 10
scaler = RobustScaler()
best_f1 = 0
final_threshold = 0.5

with tqdm(total=len(models)*tscv.n_splits, desc='Model Training') as pbar:
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        fold_scores = []
        fold_thresholds = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y)):
            # Handle imbalance only on training fold
            smote = SMOTE(sampling_strategy=0.3, random_state=fold)
            X_train, y_train = smote.fit_resample(X.iloc[train_idx], y.iloc[train_idx])
            
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Scale data
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model with massively increased iterations
            if name == 'XGBoost':
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    early_stopping_rounds=200,  # Doubled from 100
                    verbose=50
                )
            else:
                model.fit(X_train_scaled, y_train)
            
            # Threshold optimization
            if hasattr(model, "predict_proba"):
                val_probs = model.predict_proba(X_val_scaled)[:, 1]
                precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
                best_threshold = thresholds[np.argmax(f1_scores)]
                val_preds = (val_probs >= best_threshold).astype(int)
            else:
                val_preds = model.predict(X_val_scaled)
                best_threshold = 0.5
            
            fold_f1 = f1_score(y_val, val_preds)
            fold_scores.append(fold_f1)
            fold_thresholds.append(best_threshold)
            
            pbar.update(1)
            pbar.set_postfix({'Model': name, 'Fold': fold+1, 'F1': f"{fold_f1:.3f}"})
        
        mean_f1 = np.mean(fold_scores)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model = model
            final_threshold = np.median(fold_thresholds)
            print(f"New best model: {name} (F1: {mean_f1:.4f}, Threshold: {final_threshold:.4f})")

# %% [code] -- Predictions & Submission
if best_model is not None:
    # Generate predictions
    test_scaled = scaler.transform(test_final)
    
    if hasattr(best_model, "predict_proba"):
        test_probs = best_model.predict_proba(test_scaled)[:, 1]
        final_preds = (test_probs >= final_threshold).astype(int)
    else:
        final_preds = best_model.predict(test_scaled)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test['ID'].values,
        'target': final_preds.astype('int8')
    })
    
    # Validate submission
    assert len(submission) == len(test), "Submission length mismatch"
    assert list(submission.columns) == list(sample_sub.columns), "Column mismatch"
    
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission created successfully!")
    print(f"Anomaly rate: {submission['target'].mean():.4f}")
else:
    print("Model training failed - no submission generated")

# %% [code] -- Model Interpretation
def plot_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=features.columns)
        importance.nlargest(15).plot(kind='barh', title='Feature Importance')
        plt.show()
        
        # Feature correlation with target
        corr_matrix = features.join(y).corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix[['target']].sort_values('target', ascending=False), 
                    annot=True, cmap='coolwarm', center=0)
        plt.title("Feature-Target Correlation")
        plt.show()
    else:
        print("Feature importance not available for this model type")

if best_model is not None:
    plot_feature_importance(best_model, X)
    
    # SHAP values (optional)
    try:
        import shap
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X.sample(1000))
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X.sample(1000), plot_type="bar")
        plt.title("SHAP Feature Importance")
        plt.show()
    except ImportError:
        print("\nInstall SHAP for detailed interpretation: pip install shap")
    except Exception as e:
        print(f"\nSHAP error: {str(e)}")
