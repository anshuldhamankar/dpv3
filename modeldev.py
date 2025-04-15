# advanced_modeldev.py
from shared import session_data

import os
import uuid
import joblib
import pandas as pd
import traceback
from flask import Blueprint, request, jsonify, send_file
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import shap
from io import BytesIO
from datetime import datetime
from flask import Blueprint, request, jsonify
modeldev = Blueprint("modeldev", __name__)


MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Reference to global session from main.py

# Utility

def get_model_instance(name, objective):
    if objective == 'classification':
        return {
            "RandomForestClassifier": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "XGBoostClassifier": XGBClassifier(eval_metric='logloss'),
            "LGBMClassifier": LGBMClassifier(),
            "CatBoostClassifier": CatBoostClassifier(verbose=0),
            "SVC": SVC(probability=True)
        }.get(name)

    elif objective == 'regression':
        return {
            "RandomForestRegressor": RandomForestRegressor(),
            "LinearRegression": LinearRegression(),
            "XGBoostRegressor": XGBRegressor(),
            "LGBMRegressor": LGBMRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=0),
            "SVR": SVR()
        }.get(name)

    elif objective == 'clustering':
        return {
            "KMeans": KMeans(n_clusters=3, random_state=42)
        }.get(name)

    return None


@modeldev.route('/modeldev/<file_id>/columns', methods=['GET'])
def get_columns(file_id):
    try:
        df = session_data[file_id]["modified"]
        columns = list(df.columns)
        dtypes = df.dtypes.astype(str).to_dict()
        return jsonify({"columns": columns, "types": dtypes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@modeldev.route('/modeldev/<file_id>/eda', methods=['POST'])
def generate_eda(file_id):
    df = session_data[file_id]['modified']
    try:
        info = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'nulls': df.isnull().sum().to_dict(),
            'preview': df.head(5).replace({pd.NA: None}).to_dict(orient='records'),
            'summary': df.describe(include='all').replace({pd.NA: None}).to_dict()
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})


@modeldev.route('/modeldev/<file_id>/recommend', methods=['POST'])
def recommend_models(file_id):
    data = request.json
    objective = data.get("objective", "classification").lower()
    if objective == "classification":
        return jsonify({"models": ["RandomForestClassifier", "XGBoostClassifier", "LGBMClassifier", "CatBoostClassifier", "SVC"]})
    elif objective == "regression":
        return jsonify({"models": ["RandomForestRegressor", "XGBoostRegressor", "LGBMRegressor", "CatBoostRegressor", "LinearRegression"]})
    elif objective == "clustering":
        return jsonify({"models": ["KMeans"]})
    else:
        return jsonify({"error": "Unknown objective"}), 400


@modeldev.route('/modeldev/<file_id>/train', methods=['POST'])
def train_model(file_id):
    try:
        df = session_data[file_id]["modified"]
        data = request.get_json()

        objective = data['objective'].lower()
        model_name = data['model']
        target_col = data.get('target')
        selected_features = data.get('features', [])
        train_split = data['train_split'] / 100.0
        cv = int(data.get("cv", 5))

        model = get_model_instance(model_name, objective)
        if not model:
            return jsonify({"error": f"Model {model_name} not supported"}), 400

        # Validate target column
        if objective != 'clustering' and target_col not in df.columns:
            return jsonify({"error": f"Target column '{target_col}' not found in dataset"}), 400

        # Clustering (no target)
        if objective == 'clustering':
            X = df.select_dtypes(include=['number']).dropna()
            model.fit(X)
            y_pred = model.labels_
            metrics = {'clusters': y_pred.tolist()}
            scores = None

        else:
            # Validate features
            if not selected_features or not all(col in df.columns for col in selected_features):
                return jsonify({"error": "Selected features are invalid or missing"}), 400

            X = df[selected_features].dropna()
            y = df[target_col].loc[X.index]

            X = pd.get_dummies(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1 - train_split, random_state=42)

            model.fit(X_train, y_train)
            scores = cross_val_score(model, X_train, y_train, cv=cv)

            y_pred = model.predict(X_test)
            metrics = {}

            if objective == "classification":
                metrics['report'] = classification_report(y_test, y_pred, output_dict=True)
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                except:
                    metrics['roc_auc'] = None
            elif objective == "regression":
                metrics['mse'] = mean_squared_error(y_test, y_pred)

        # Save model
        filename = f"{file_id}_{model_name}.pkl"
        path = os.path.join(MODEL_DIR, filename)
        joblib.dump(model, path)

        return jsonify({
            "success": True,
            "cv_score": scores.mean() if scores is not None else None,
            "metrics": metrics,
            "model_path": filename,
            "model": model_name
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
