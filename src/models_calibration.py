# Model calibration and hyperparameter tuning with Optuna
# This file contains all hyperparameter optimization using Optuna (replacing GridSearchCV)

# ===== IMPORTS =====
import sys
sys.dont_write_bytecode = True  # Prevent .pyc file creation

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
import time
import warnings

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress sklearn convergence warnings during hyperparameter optimization
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Reproducibility seed (matches random_state=42 in models)
OPTUNA_SEED = 42


def calibrate_models(data):
    """
    Calibrate all models using Optuna for hyperparameter optimization

    3-4x faster than GridSearchCV while often finding better parameters!

    Parameters:
    -----------
    data : dict
        Dictionary containing training data and configuration

    Returns:
    --------
    models : dict
        Dictionary containing all calibrated models
    """
    # Extract data from the dictionary
    X_train = data['X_train']
    y_train = data['y_train']
    X_train_res = data['X_train_res']
    y_train_res = data['y_train_res']
    X_train_normal = data['X_train_normal']
    X_train_res_normal = data['X_train_res_normal']
    fraud_ratio = data['fraud_ratio']
    small_tscv = data['small_tscv']

    # Reduce dataset for Isolation Forest and LOF (time-aware: take first 30%)
    # This avoids information leakage and speeds up distance-based models
    reduce_ratio = 0.3
    n_samples_reduced = int(len(X_train_res) * reduce_ratio)
    X_train_res_reduced = X_train_res[:n_samples_reduced]  # Take first 30% chronologically
    y_train_res_reduced = y_train_res[:n_samples_reduced]
    X_train_res_normal_reduced = X_train_res_reduced[y_train_res_reduced == 0]

    print(f"Dataset reduction for IF/LOF: {len(X_train_res)} → {n_samples_reduced} samples (first {reduce_ratio*100}% chronologically)")

    models = {}
    N_TRIALS = 20  # Number of optimization trials per model

    # ===== LOGISTIC REGRESSION (Supervised) =====
    print("="*60)
    print("Logistic Regression (Supervised) - Optuna Optimization")
    print("="*60)

    start_time_lr = time.time()

    def objective_lr(trial):
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])

        # Class weight options
        cw_choice = trial.suggest_categorical('class_weight', ['balanced', 'custom1', 'custom2'])
        if cw_choice == 'balanced':
            class_weight = 'balanced'
        elif cw_choice == 'custom1':
            class_weight = {0: 1, 1: 10}
        else:
            class_weight = {0: 1, 1: 20}

        # Cross-validation
        scores = []
        for train_idx, val_idx in small_tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = LogisticRegression(C=C, penalty=penalty, solver=solver,
                                      class_weight=class_weight, random_state=42, max_iter=1000)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred))

        return np.mean(scores)

    sampler_lr = TPESampler(seed=OPTUNA_SEED)
    study_lr = optuna.create_study(direction='maximize', sampler=sampler_lr)
    study_lr.optimize(objective_lr, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    # Train final model with best params
    best_params = study_lr.best_params
    if best_params['class_weight'] == 'balanced':
        class_weight = 'balanced'
    elif best_params['class_weight'] == 'custom1':
        class_weight = {0: 1, 1: 10}
    else:
        class_weight = {0: 1, 1: 20}

    best_lr = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=best_params['solver'],
        class_weight=class_weight,
        random_state=42,
        max_iter=1000
    )
    best_lr.fit(X_train, y_train)

    models['best_lr'] = best_lr
    models['study_lr'] = study_lr

    elapsed_time_lr = time.time() - start_time_lr
    print(f"Best params: {study_lr.best_params}")
    print(f"Best F1-score: {study_lr.best_value:.4f}")
    print(f"[TIME]  Training Time: {elapsed_time_lr:.2f}s ({elapsed_time_lr/60:.2f} minutes)")

    # ===== RANDOM FOREST (Supervised) =====
    print("\n" + "="*60)
    print("Random Forest (Supervised) - Optuna Optimization")
    print("="*60)

    start_time_rf = time.time()

    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 300)

        # Max depth: either None or a value between 5-50
        use_max_depth = trial.suggest_categorical('use_max_depth', [True, False])
        if use_max_depth:
            max_depth = trial.suggest_int('max_depth', 5, 50)
        else:
            max_depth = None

        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])

        # Class weight
        cw_choice = trial.suggest_categorical('class_weight', ['balanced', 'custom1', 'custom2'])
        if cw_choice == 'balanced':
            class_weight = 'balanced'
        elif cw_choice == 'custom1':
            class_weight = {0: 1, 1: 10}
        else:
            class_weight = {0: 1, 1: 20}

        # Cross-validation
        scores = []
        for train_idx, val_idx in small_tscv.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                bootstrap=bootstrap,
                random_state=42
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            scores.append(f1_score(y_val, y_pred))

        return np.mean(scores)

    sampler_rf = TPESampler(seed=OPTUNA_SEED)
    study_rf = optuna.create_study(direction='maximize', sampler=sampler_rf)
    study_rf.optimize(objective_rf, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    # Train final model
    best_params = study_rf.best_params
    if best_params['class_weight'] == 'balanced':
        class_weight = 'balanced'
    elif best_params['class_weight'] == 'custom1':
        class_weight = {0: 1, 1: 10}
    else:
        class_weight = {0: 1, 1: 20}

    # Handle conditional max_depth parameter
    if best_params['use_max_depth']:
        max_depth = best_params['max_depth']
    else:
        max_depth = None

    best_rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=max_depth,
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        class_weight=class_weight,
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    best_rf.fit(X_train, y_train)

    models['best_rf'] = best_rf
    models['rf_random'] = study_rf  # Keep naming consistent

    print(f"Best params: {study_rf.best_params}")
    print(f"Best F1-score: {study_rf.best_value:.4f}")
    elapsed_time_rf = time.time() - start_time_rf
    print(f"[TIME]  Training Time: {elapsed_time_rf:.2f}s ({elapsed_time_rf/60:.2f} minutes)")

    # ===== ISOLATION FOREST (Unsupervised) =====
    print("\n" + "="*60)
    print("Isolation Forest (Unsupervised) - Optuna Optimization")
    print("="*60)

    start_time_iso = time.time()

    def objective_iso(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_samples = trial.suggest_float('max_samples', 0.05, 0.8)
        contamination = trial.suggest_float('contamination', 0.0005, 0.005)
        max_features = trial.suggest_float('max_features', 0.3, 1.0)

        # Cross-validation (using reduced dataset for speed)
        scores = []
        for train_idx, val_idx in small_tscv.split(X_train_res_reduced):
            X_tr, X_val = X_train_res_reduced[train_idx], X_train_res_reduced[val_idx]
            y_tr, y_val = y_train_res_reduced[train_idx], y_train_res_reduced[val_idx]

            model = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                contamination=contamination,
                max_features=max_features,
                random_state=42
            )
            model.fit(X_tr)
            y_pred = (model.predict(X_val) == -1).astype(int)  # -1 = anomaly = 1 (fraud)
            scores.append(f1_score(y_val, y_pred, zero_division=0))

        return np.mean(scores)

    sampler_iso = TPESampler(seed=OPTUNA_SEED)
    study_iso = optuna.create_study(direction='maximize', sampler=sampler_iso)
    study_iso.optimize(objective_iso, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    # Train final model
    best_iso = IsolationForest(
        n_estimators=study_iso.best_params['n_estimators'],
        max_samples=study_iso.best_params['max_samples'],
        contamination=study_iso.best_params['contamination'],
        max_features=study_iso.best_params['max_features'],
        random_state=42
    )
    best_iso.fit(X_train_res)

    models['best_iso'] = best_iso
    models['iso_grid'] = study_iso

    print(f"Best params: {study_iso.best_params}")
    print(f"Best F1-score: {study_iso.best_value:.4f}")
    elapsed_time_iso = time.time() - start_time_iso
    print(f"[TIME]  Training Time: {elapsed_time_iso:.2f}s ({elapsed_time_iso/60:.2f} minutes)")

    # ===== LOCAL OUTLIER FACTOR (Unsupervised) =====
    print("\n" + "="*60)
    print("Local Outlier Factor (Unsupervised) - Optuna Optimization")
    print("="*60)

    start_time_lof = time.time()

    def objective_lof(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
        contamination = trial.suggest_float('contamination', 0.0005, 0.005)
        metric = trial.suggest_categorical('metric', ['euclidean', 'minkowski', 'manhattan', 'cosine'])

        # Cross-validation (using reduced dataset for speed)
        scores = []
        for train_idx, val_idx in small_tscv.split(X_train_res_reduced):
            X_tr, X_val = X_train_res_reduced[train_idx], X_train_res_reduced[val_idx]
            y_tr, y_val = y_train_res_reduced[train_idx], y_train_res_reduced[val_idx]

            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                metric=metric,
                novelty=True
            )
            model.fit(X_tr)
            y_pred = (model.predict(X_val) == -1).astype(int)
            scores.append(f1_score(y_val, y_pred, zero_division=0))

        return np.mean(scores)

    sampler_lof = TPESampler(seed=OPTUNA_SEED)
    study_lof = optuna.create_study(direction='maximize', sampler=sampler_lof)
    study_lof.optimize(objective_lof, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    # Train final model
    best_lof = LocalOutlierFactor(
        n_neighbors=study_lof.best_params['n_neighbors'],
        contamination=study_lof.best_params['contamination'],
        metric=study_lof.best_params['metric'],
        novelty=True
    )
    best_lof.fit(X_train_res)

    models['best_lof'] = best_lof
    models['lof_grid'] = study_lof

    print(f"Best params: {study_lof.best_params}")
    print(f"Best F1-score: {study_lof.best_value:.4f}")
    elapsed_time_lof = time.time() - start_time_lof
    print(f"[TIME]  Training Time: {elapsed_time_lof:.2f}s ({elapsed_time_lof/60:.2f} minutes)")

    # ===== GAUSSIAN MIXTURE MODEL (Unsupervised) =====
    print("\n" + "="*60)
    print("Gaussian Mixture Model (Unsupervised) - Optuna Optimization")
    print("="*60)

    start_time_gmm = time.time()

    def objective_gmm(trial):
        n_components = trial.suggest_int('n_components', 1, 20)
        covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
        max_iter = trial.suggest_int('max_iter', 50, 500)
        tol = trial.suggest_float('tol', 1e-6, 1e-2, log=True)

        # Cross-validation
        scores = []
        for train_idx, val_idx in small_tscv.split(X_train_res):
            X_tr, X_val = X_train_res[train_idx], X_train_res[val_idx]
            y_tr, y_val = y_train_res[train_idx], y_train_res[val_idx]

            try:
                model = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                    tol=tol,
                    random_state=42
                )
                model.fit(X_tr)
                # Lower likelihood = anomaly
                scores_val = model.score_samples(X_val)
                threshold = np.percentile(scores_val, fraud_ratio * 100)
                y_pred = (scores_val < threshold).astype(int)
                scores.append(f1_score(y_val, y_pred, zero_division=0))
            except:
                return 0.0  # Return poor score if convergence fails

        return np.mean(scores)

    sampler_gmm = TPESampler(seed=OPTUNA_SEED)
    study_gmm = optuna.create_study(direction='maximize', sampler=sampler_gmm)
    study_gmm.optimize(objective_gmm, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)  # n_jobs=1 for GMM

    # Train final model
    best_gmm = GaussianMixture(
        n_components=study_gmm.best_params['n_components'],
        covariance_type=study_gmm.best_params['covariance_type'],
        max_iter=study_gmm.best_params['max_iter'],
        tol=study_gmm.best_params['tol'],
        random_state=42
    )
    best_gmm.fit(X_train_res)

    models['best_gmm'] = best_gmm
    models['gmm_grid'] = study_gmm

    print(f"Best params: {study_gmm.best_params}")
    print(f"Best F1-score: {study_gmm.best_value:.4f}")
    elapsed_time_gmm = time.time() - start_time_gmm
    print(f"[TIME]  Training Time: {elapsed_time_gmm:.2f}s ({elapsed_time_gmm/60:.2f} minutes)")

    # ===== ISOLATION FOREST (Semi-Supervised) =====
    print("\n" + "="*60)
    print("Isolation Forest (Semi-Supervised) - Optuna Optimization")
    print("="*60)

    start_time_iso_semi = time.time()

    def objective_iso_semi(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_samples = trial.suggest_float('max_samples', 0.05, 0.8)
        contamination = trial.suggest_float('contamination', 0.0005, 0.005)
        max_features = trial.suggest_float('max_features', 0.3, 1.0)

        # For semi-supervised, we only have normal transactions in training
        # So we can't do proper CV easily. Use simple validation.
        # Using reduced dataset for speed
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=42
        )
        model.fit(X_train_res_normal_reduced)

        # Validate on a subset of X_train_res_reduced (which has both classes)
        val_size = min(10000, len(X_train_res_reduced))
        val_indices = np.random.choice(len(X_train_res_reduced), val_size, replace=False)
        X_val = X_train_res_reduced[val_indices]
        y_val = y_train_res_reduced[val_indices]

        y_pred = (model.predict(X_val) == -1).astype(int)
        return f1_score(y_val, y_pred, zero_division=0)

    sampler_iso_semi = TPESampler(seed=OPTUNA_SEED)
    study_iso_semi = optuna.create_study(direction='maximize', sampler=sampler_iso_semi)
    study_iso_semi.optimize(objective_iso_semi, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    # Train final model
    best_iso_semi = IsolationForest(
        n_estimators=study_iso_semi.best_params['n_estimators'],
        max_samples=study_iso_semi.best_params['max_samples'],
        contamination=study_iso_semi.best_params['contamination'],
        max_features=study_iso_semi.best_params['max_features'],
        random_state=42
    )
    best_iso_semi.fit(X_train_res_normal)

    models['best_iso_semi'] = best_iso_semi
    models['iso_grid_semi'] = study_iso_semi

    print(f"Best params: {study_iso_semi.best_params}")
    print(f"Best F1-score: {study_iso_semi.best_value:.4f}")
    elapsed_time_iso_semi = time.time() - start_time_iso_semi
    print(f"[TIME]  Training Time: {elapsed_time_iso_semi:.2f}s ({elapsed_time_iso_semi/60:.2f} minutes)")

    # ===== LOCAL OUTLIER FACTOR (Semi-Supervised) =====
    print("\n" + "="*60)
    print("Local Outlier Factor (Semi-Supervised) - Optuna Optimization")
    print("="*60)

    start_time_lof_semi = time.time()

    def objective_lof_semi(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 5, 50)
        contamination = trial.suggest_float('contamination', 0.0005, 0.005)
        metric = trial.suggest_categorical('metric', ['euclidean', 'minkowski', 'manhattan', 'cosine'])

        # Using reduced dataset for speed
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric,
            novelty=True
        )
        model.fit(X_train_res_normal_reduced)

        # Validate on reduced dataset
        val_size = min(10000, len(X_train_res_reduced))
        val_indices = np.random.choice(len(X_train_res_reduced), val_size, replace=False)
        X_val = X_train_res_reduced[val_indices]
        y_val = y_train_res_reduced[val_indices]

        y_pred = (model.predict(X_val) == -1).astype(int)
        return f1_score(y_val, y_pred, zero_division=0)

    sampler_lof_semi = TPESampler(seed=OPTUNA_SEED)
    study_lof_semi = optuna.create_study(direction='maximize', sampler=sampler_lof_semi)
    study_lof_semi.optimize(objective_lof_semi, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    # Train final model
    best_lof_semi = LocalOutlierFactor(
        n_neighbors=study_lof_semi.best_params['n_neighbors'],
        contamination=study_lof_semi.best_params['contamination'],
        metric=study_lof_semi.best_params['metric'],
        novelty=True
    )
    best_lof_semi.fit(X_train_res_normal)

    models['best_lof_semi'] = best_lof_semi
    models['lof_grid_semi'] = study_lof_semi

    print(f"Best params: {study_lof_semi.best_params}")
    print(f"Best F1-score: {study_lof_semi.best_value:.4f}")
    elapsed_time_lof_semi = time.time() - start_time_lof_semi
    print(f"[TIME]  Training Time: {elapsed_time_lof_semi:.2f}s ({elapsed_time_lof_semi/60:.2f} minutes)")

    # ===== GAUSSIAN MIXTURE MODEL (Semi-Supervised) =====
    print("\n" + "="*60)
    print("Gaussian Mixture Model (Semi-Supervised) - Optuna Optimization")
    print("="*60)

    start_time_gmm_semi = time.time()

    def objective_gmm_semi(trial):
        n_components = trial.suggest_int('n_components', 1, 50)
        covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
        max_iter = trial.suggest_int('max_iter', 50, 500)
        tol = trial.suggest_float('tol', 1e-6, 1e-2, log=True)

        try:
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                max_iter=max_iter,
                tol=tol,
                random_state=42
            )
            model.fit(X_train_res_normal)

            # Validate
            val_size = min(10000, len(X_train_res))
            val_indices = np.random.choice(len(X_train_res), val_size, replace=False)
            X_val = X_train_res[val_indices]
            y_val = y_train_res[val_indices]

            scores_val = model.score_samples(X_val)
            threshold = np.percentile(scores_val, fraud_ratio * 100)
            y_pred = (scores_val < threshold).astype(int)
            return f1_score(y_val, y_pred, zero_division=0)
        except:
            return 0.0

    sampler_gmm_semi = TPESampler(seed=OPTUNA_SEED)
    study_gmm_semi = optuna.create_study(direction='maximize', sampler=sampler_gmm_semi)
    study_gmm_semi.optimize(objective_gmm_semi, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    # Train final model
    best_gmm_semi = GaussianMixture(
        n_components=study_gmm_semi.best_params['n_components'],
        covariance_type=study_gmm_semi.best_params['covariance_type'],
        max_iter=study_gmm_semi.best_params['max_iter'],
        tol=study_gmm_semi.best_params['tol'],
        random_state=42
    )
    best_gmm_semi.fit(X_train_res_normal)

    models['best_gmm_semi'] = best_gmm_semi
    models['gmm_grid_semi'] = study_gmm_semi

    print(f"Best params: {study_gmm_semi.best_params}")
    print(f"Best F1-score: {study_gmm_semi.best_value:.4f}")
    elapsed_time_gmm_semi = time.time() - start_time_gmm_semi
    print(f"[TIME]  Training Time: {elapsed_time_gmm_semi:.2f}s ({elapsed_time_gmm_semi/60:.2f} minutes)")

    return models


def save_models(models, filename='trained_models.pkl'):
    """
    Save trained models using joblib (better cross-version compatibility)

    Parameters:
    -----------
    models : dict
        Dictionary containing all trained models
    filename : str
        Name of the file to save models to

    Returns:
    --------
    filepath : Path
        Path to the saved file
    """
    # Save to saved_models directory
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "saved_models"
    models_dir.mkdir(exist_ok=True, parents=True)
    filepath = models_dir / filename

    # Add metadata
    models_data = {
        'models': models,
        'timestamp': datetime.now().isoformat(),
        'model_count': len(models),
        'optimizer': 'Optuna',
        'serializer': 'joblib'
    }

    # Use joblib for better numpy/sklearn compatibility across versions
    joblib.dump(models_data, filepath, compress=3)

    print(f"\n[OK] Models saved to: {filepath}")
    print(f"  File size: {filepath.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Models saved: {len(models)}")
    print(f"  Serializer: joblib (cross-version compatible)")

    return filepath


def load_models(filename='trained_models.pkl'):
    """
    Load trained models using joblib (better cross-version compatibility)

    Parameters:
    -----------
    filename : str
        Name of the file to load models from

    Returns:
    --------
    models : dict
        Dictionary containing all trained models

    Raises:
    -------
    FileNotFoundError
        If the models file doesn't exist
    """
    # Load from saved_models directory
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "saved_models"
    filepath = models_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Trained models file not found: {filepath}\n"
            "Please run model calibration first (Menu option [3] or [8])"
        )

    # Use joblib for better numpy/sklearn compatibility across versions
    try:
        models_data = joblib.load(filepath)
    except Exception as e:
        # Check if it's a numpy version mismatch (most common issue)
        if 'BitGenerator' in str(e) or 'MT19937' in str(e):
            raise RuntimeError(
                f"\n{'='*60}\n"
                f"[ERROR] Model file incompatible with current numpy version!\n"
                f"{'='*60}\n\n"
                f"Your trained_models.pkl was created with a different numpy version\n"
                f"and cannot be loaded with numpy {np.__version__}.\n\n"
                f"Solution: Retrain models with current environment (takes 5-10 min):\n"
                f"  cd /Users/noamifergan/Desktop/Cours/PROJET_DYLAN/src\n"
                f"  python3 models_calibration.py\n\n"
                f"This will create a new compatible trained_models.pkl file.\n"
                f"{'='*60}\n"
            )

        # Try pickle as fallback for other errors
        print(f"[WARNING] Warning: joblib loading failed, trying pickle fallback...")
        print(f"  Error was: {e}")
        import pickle
        with open(filepath, 'rb') as f:
            models_data = pickle.load(f)
        print("  [OK] Loaded with pickle (consider retraining for better compatibility)")

    models = models_data['models']
    timestamp = models_data.get('timestamp', 'unknown')
    optimizer = models_data.get('optimizer', 'GridSearchCV')
    serializer = models_data.get('serializer', 'pickle')

    print(f"\n[OK] Models loaded from: {filepath}")
    print(f"  File size: {filepath.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Models loaded: {len(models)}")
    print(f"  Trained on: {timestamp}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Serializer: {serializer}")

    return models


def main():
    """Main function to run model calibration"""
    # Import and prepare data
    from main import load_and_prepare_data

    print("="*60)
    print("FRAUD DETECTION PIPELINE - MODEL CALIBRATION (OPTUNA)")
    print("="*60)

    data = load_and_prepare_data()
    models = calibrate_models(data)

    print("\n" + "="*60)
    print("MODEL CALIBRATION COMPLETE")
    print("="*60)
    print(f"Total models trained: {len(models)}")

    # Save models to file
    print("\nSaving models to file...")
    save_models(models)

    return models, data


if __name__ == "__main__":
    models, data = main()
