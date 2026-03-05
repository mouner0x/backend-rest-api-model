import os
import pandas as pd
import joblib
from datetime import datetime, timezone
from fastapi import HTTPException
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ..database import db

MODELS_STORAGE_DIR = "models"

ALLOWED_MODELS = [
    "random_forest",
    "linear_regression",
    "svm",
    "xgboost",
    "neural_network"
]

# =========================================================
# CREATE MODEL
# =========================================================

async def create_selected_model(data: dict, current_user: Dict[str, Any]) -> dict:
    from bson import ObjectId

    dataset_id = data.get("dataset_id")
    model_type = data.get("model_type")

    if model_type not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    user_id = current_user.get("_id")
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    obj_dataset_id = ObjectId(dataset_id)

    dataset = await db.datasets.find_one({
        "_id": obj_dataset_id,
        "user_id": user_id
    })

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    model_name = f"{model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    model_doc = {
        "user_id": user_id,
        "dataset_id": obj_dataset_id,
        "dataset_name": dataset.get("filename", "Unknown Dataset"),
        "model_name": model_name,
        "model_type": model_type,
        "status": "selected",
        "metrics": None,
        "model_path": None,
        "target_column": dataset.get("target_column"),
        "training_features": [],
        "created_at": datetime.now(timezone.utc),
        "trained_at": None
    }

    result = await db.models.insert_one(model_doc)

    return {
        "model_id": str(result.inserted_id),
        "model_type": model_type,
        "status": "selected"
    }

# =========================================================
# TRAIN MODEL
# =========================================================

async def train_model_service(model_id: str) -> dict:
    from bson import ObjectId

    obj_model_id = ObjectId(model_id)

    model_doc = await db.models.find_one({"_id": obj_model_id})
    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found")

    dataset_doc = await db.datasets.find_one({"_id": model_doc.get("dataset_id")})
    if not dataset_doc:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset_doc.get("file_path"))

    target_column = model_doc.get("target_column")
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column missing")

    df = df.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    training_features = X.columns.tolist()

    numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    model_type = model_doc.get("model_type")

    if model_type == "random_forest":
        estimator = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == "linear_regression":
        estimator = LinearRegression()
    elif model_type == "svm":
        estimator = SVR()
    elif model_type == "neural_network":
        estimator = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    elif model_type == "xgboost":
        estimator = XGBRegressor(n_estimators=200, random_state=42)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model type")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mse = float(mean_squared_error(y_test, predictions))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    os.makedirs(MODELS_STORAGE_DIR, exist_ok=True)
    model_file_path = f"{MODELS_STORAGE_DIR}/{model_id}.pkl"
    joblib.dump(pipeline, model_file_path)

    await db.models.update_one(
        {"_id": obj_model_id},
        {"$set": {
            "status": "trained",
            "training_features": training_features,
            "metrics": {
                "mse": mse,
                "mae": mae,
                "r2_score": r2
            },
            "model_path": model_file_path,
            "trained_at": datetime.now(timezone.utc)
        }}
    )

    return {
        "mse": mse,
        "mae": mae,
        "r2_score": r2,
        "status": "trained"
    }

# =========================================================
# PREDICT (UPDATED WITH FEATURE IMPORTANCE)
# =========================================================

async def predict_pipeline_service(model_id: str, input_data: dict):
    from bson import ObjectId

    obj_model_id = ObjectId(model_id)

    model_doc = await db.models.find_one({"_id": obj_model_id})
    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found")

    if model_doc.get("status") != "trained":
        raise HTTPException(status_code=400, detail="Model not trained")

    model_path = model_doc.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model file missing")

    pipeline = joblib.load(model_path)

    training_features = model_doc.get("training_features")
    if not training_features:
        raise HTTPException(status_code=500, detail="Missing training_features metadata")

    normalized_input = {
        key.strip().lower().replace(" ", "_"): value
        for key, value in input_data.items()
    }

    missing = [f for f in training_features if f not in normalized_input]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}"
        )

    ordered_input = {f: normalized_input[f] for f in training_features}

    input_df = pd.DataFrame([ordered_input])

    prediction = pipeline.predict(input_df)[0]

    # ===============================
    # FEATURE IMPORTANCE FOR CHART
    # ===============================

    feature_importance = []

    try:
        model = pipeline.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            for i, feature in enumerate(training_features):
                feature_importance.append({
                    "feature": feature,
                    "impact": float(importances[i])
                })

    except Exception:
        feature_importance = []

    return {
        "prediction": float(prediction),
        "feature_importance": feature_importance
    }

# =========================================================
# GET MODEL
# =========================================================

async def get_model_service(model_id: str, current_user: Dict[str, Any]) -> dict:
    from bson import ObjectId

    obj_model_id = ObjectId(model_id)

    model_doc = await db.models.find_one({
        "_id": obj_model_id,
        "user_id": current_user.get("_id")
    })

    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "success": True,
        "message": "Model retrieved successfully",
        "data": {
            "model_id": str(model_doc.get("_id")),
            "model_name": model_doc.get("model_name"),
            "training_features": model_doc.get("training_features"),
            "model_type": model_doc.get("model_type"),
            "dataset_id": str(model_doc.get("dataset_id")),
            "dataset_name": model_doc.get("dataset_name"),
            "target_column": model_doc.get("target_column"),
            "r2_score": model_doc.get("metrics", {}).get("r2_score"),
            "status": model_doc.get("status"),
            "created_at": model_doc.get("created_at")
        }
    }