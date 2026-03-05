import os
import time
import logging
import math
import pandas as pd
import joblib
from datetime import datetime, timezone
from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
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

MODELS_STORAGE_DIR = "models"
logger = logging.getLogger(__name__)

async def train_model_service(model_id: str, current_user: Dict[str, Any], db: AsyncIOMotorDatabase) -> dict:
    from bson import ObjectId
    user_id = current_user["_id"]
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            pass
            
    try:
        obj_model_id = ObjectId(model_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid model ID format")

    model_doc = await db.models.find_one({"_id": obj_model_id, "user_id": user_id})
    if not model_doc:
        raise HTTPException(status_code=404, detail="Model not found or you don't have access")
        
    if model_doc.get("status") != "selected":
        raise HTTPException(status_code=400, detail="Model is not in 'selected' status and cannot be trained")

    dataset_id = model_doc.get("dataset_id")
    dataset_doc = await db.datasets.find_one({"_id": dataset_id, "user_id": user_id})
    
    if not dataset_doc:
        raise HTTPException(status_code=404, detail="Associated dataset not found")

    file_path = dataset_doc.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Dataset file missing on disk")

    # 1) Safe CSV Reading (Encoding Handling)
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding="cp1256")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading dataset with fallback encoding: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

    rows_before, cols_before = df.shape
    start_time = time.time()

    try:
        # Preprocessing block
        
        # 8) ADD DEBUG LOGS
        original_shape = df.shape
        print("Shape before encoding:", original_shape)
        
        # 2) Automatic Dataset Cleaning: DROP ID-LIKE COLUMNS
        id_cols = [c for c in df.columns if "id" in c.lower() or "ID" in c]
        if id_cols:
            df = df.drop(columns=[c for c in id_cols if c != target_column], errors="ignore")

        target_column = model_doc.get("target_column", dataset_doc.get("target_column"))
        
        # 7) Target Column Validation (Initial)
        if not target_column or target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Target column is missing or not set correctly")

        # 1) CLEAN SPECIAL TEXT COLUMNS
        for col in df.columns:
            if col == target_column: continue
            if df[col].dtype == "object":
                col_str = df[col].astype(str)
                # Check for "km"
                if col_str.str.contains(" km", na=False).any():
                    df[col] = col_str.str.replace(" km", "", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                # Check for date-like columns (e.g. "4-May")
                elif col_str.str.contains(r"^\d{1,2}-[a-zA-Z]{3}$|^[a-zA-Z]{3}-\d{1,2}$", regex=True, na=False).any():
                    # Convert to string only (do NOT one-hot blindly later, we handle via LabelEncoding)
                    df[col] = df[col].astype(str)

        # Preliminary string cleaning
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(["-", "", "None", "nan", "NaN"], None)
                df[col] = df[col].str.replace(",", "", regex=False)

        # 2) Drop avg string len > 100 or unique value ratio > 0.9
                if col != target_column:
                    valid_strs = df[col].dropna()
                    if len(valid_strs) > 0:
                        if valid_strs.str.len().mean() > 100:
                            df = df.drop(columns=[col])
                            continue
                    
                    # unique value ratio > 0.9
                    if len(df) > 0 and (df[col].nunique() / len(df)) > 0.9:
                        df = df.drop(columns=[col])
                        continue

                # 4) Normalize Boolean Columns
                lower_vals = df[col].dropna().astype(str).str.lower()
                if len(lower_vals) > 0 and lower_vals.isin(["true", "false", "yes", "no", "1", "0"]).all():
                    bool_map = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
                    df[col] = df[col].map(lambda x: bool_map.get(str(x).lower(), x) if pd.notna(x) else x)
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3) Numeric Cleaning (Extract first, then convert)
        import re
        for col in df.columns:
            if col == target_column:
                # 7) Target Column Validation - ensure numeric
                df[col] = pd.to_numeric(df[col], errors="coerce")
                continue
                
            if df[col].dtype == "object":
                # extract numeric part using regex
                extracted = df[col].astype(str).apply(lambda x: re.search(r"[-+]?\d*\.\d+|\d+", x).group() if pd.notna(x) and re.search(r"[-+]?\d*\.\d+|\d+", x) else x)
                converted = pd.to_numeric(extracted, errors="coerce")
                
                non_na_before = df[col].notna().sum()
                non_na_after = converted.notna().sum()
                # Use it if it's primarily numeric
                if non_na_before > 0 and non_na_after > non_na_before * 0.5:
                    df[col] = converted
                else:
                    # fallback to standard pd.to_numeric ignore
                    pd.to_numeric(df[col], errors="ignore")

        # 6) HANDLE MISSING VALUES
        import numpy as np
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))

        cat_cols_to_fill = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols_to_fill:
            mode_s = df[col].mode()
            if not mode_s.empty:
                df[col] = df[col].fillna(mode_s.iloc[0])
            else:
                df[col] = df[col].fillna("Missing")

        # 7) Target Column Validation (Final check)
        df = df.dropna(subset=[target_column])

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Target column is completely empty or non-numeric after cleaning")
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            raise HTTPException(status_code=400, detail="Target column must be numeric")

        X = df.drop(columns=[target_column])
        training_features = X.columns.tolist()
        y = df[target_column]
        
        # 9) ENSURE TRAINING NEVER HANGS / Final Safety Check
        if X.empty or y.empty:
            raise HTTPException(status_code=400, detail="Features or target are empty after preprocessing")
        if X.isna().any().any():
            raise HTTPException(status_code=400, detail="X contains NaN values after preprocessing")
        if y.isna().any():
            raise HTTPException(status_code=400, detail="y contains NaN values after preprocessing")

        # 3) HANDLE NUMERIC DETECTION PROPERLY
        numeric_columns = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # 4) SMART CATEGORICAL ENCODING
        from sklearn.preprocessing import OrdinalEncoder
        ohe_cols = []
        le_cols = []
        for col in categorical_columns:
            unique_count = X[col].nunique()
            if unique_count <= 15:
                ohe_cols.append(col)
            else:
                le_cols.append(col)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_columns),
                ("cat_ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
                ("cat_le", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), le_cols),
            ]
        )

        model_type = model_doc.get("model_type")
        
        # 7) REDUCE MODEL COMPLEXITY
        if model_type == "random_forest":
            estimator = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
        elif model_type == "linear_regression":
            estimator = LinearRegression(n_jobs=-1)
        elif model_type == "svm":
            estimator = SVR()
        elif model_type == "neural_network":
            estimator = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
        elif model_type == "xgboost":
            estimator = XGBRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Evaluate Preprocessor dimensions
        pipeline_prep = Pipeline([("preprocessor", preprocessor)])
        X_train_prep = pipeline_prep.fit_transform(X_train)
        X_test_prep = pipeline_prep.transform(X_test)
        
        import scipy
        if scipy.sparse.issparse(X_train_prep):
            X_train_dense = X_train_prep.toarray()
            X_test_dense = X_test_prep.toarray()
        else:
            X_train_dense = X_train_prep
            X_test_dense = X_test_prep            

        # 5) LIMIT MAX FEATURES (Using SelectKBest)
        feature_count = X_train_dense.shape[1]
        
        # Equivalent functionally to:
        # correlations = df.corr(numeric_only=True)[target_column].abs()
        # top_features = correlations.sort_values(ascending=False).head(200).index
        # df = df[top_features]
        if feature_count > 200:
            from sklearn.feature_selection import SelectKBest, f_regression
            selector = SelectKBest(score_func=f_regression, k=200)
            X_train_dense = selector.fit_transform(X_train_dense, y_train)
            X_test_dense = selector.transform(X_test_dense)
            
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("selector", selector),
                ("model", estimator)
            ])
            feature_count = 200
        else:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator)
            ])

        print("Shape after encoding:", (X.shape[0], feature_count))
        
        pipeline.named_steps["model"].fit(X_train_dense, y_train)
        predictions = pipeline.named_steps["model"].predict(X_test_dense)

        cols_after_prep = feature_count

        train_time = time.time() - start_time
        
        logger.info(f"--- Training Logs for Model {model_id} ---")
        logger.info(f"Rows logged before: {rows_before}")
        logger.info(f"Columns before cleaning: {cols_before}")
        logger.info(f"Columns after preprocessing: {cols_after_prep + 1}")
        logger.info(f"Final feature count after encoding: {feature_count}")
        logger.info(f"Training time: {train_time:.2f}s")
        print(f"Model {model_id} trained in {train_time:.2f}s. Initial columns: {cols_before}. Features used: {cols_after_prep}. Final feature count: {feature_count}.")

    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        # 8) Prevent Server Freezing - catch all
        import traceback
        logger.error(f"Training failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing or training: {str(e)}")

    raw_mse = float(mean_squared_error(y_test, predictions))
    raw_mae = float(mean_absolute_error(y_test, predictions))
    raw_r2 = float(r2_score(y_test, predictions))

    mse = raw_mse if not math.isnan(raw_mse) and not math.isinf(raw_mse) else 0.0
    mae = raw_mae if not math.isnan(raw_mae) and not math.isinf(raw_mae) else 0.0
    r2 = raw_r2 if not math.isnan(raw_r2) and not math.isinf(raw_r2) else 0.0

    os.makedirs(MODELS_STORAGE_DIR, exist_ok=True)
    model_file_path = f"models/{model_id}.pkl"
    try:
        joblib.dump(pipeline, model_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving pipeline to disk: {str(e)}")

    safe_r2 = float(r2) if r2 is not None and not math.isnan(r2) else 0.0
    safe_mae = float(mae) if mae is not None and not math.isnan(mae) else 0.0
    safe_mse = float(mse) if mse is not None and not math.isnan(mse) else 0.0

    try:
        await db.models.update_one(
            {"_id": obj_model_id},
            {
                "$set": {
                    "training_features": training_features,
                    "status": "trained",
                    "user_id": user_id,
                    "metrics": {
                        "r2_score": safe_r2,
                        "mae": safe_mae,
                        "mse": safe_mse
                    },
                    "model_path": model_file_path,
                    "trained_at": datetime.now(timezone.utc)
                }
            }
        )
        print("MODEL UPDATED IN DATABASE SUCCESSFULLY")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database update failed: {str(e)}")

    response = {
        "mse": mse,
        "mae": mae,
        "r2_score": r2,
        "status": "trained"
    }

    # 4) Add final safety check before returning
    from app.utils.json_sanitizer import sanitize_for_json
    response = sanitize_for_json(response)
    
    return response
