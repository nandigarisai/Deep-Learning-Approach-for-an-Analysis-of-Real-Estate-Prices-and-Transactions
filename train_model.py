"""Train and save the price prediction models.

Run this script after changing the dataset:
    py train_model.py
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential

from config import (
    CATEGORICAL_COLUMNS,
    DATA_DIRECTORY,
    FEATURE_COLUMNS,
    MODELS_DIRECTORY,
)

TARGET_COLUMN = "Price"
RANDOM_STATE = 42


def load_training_data():
    """Load valid rows from the real-estate dataset."""
    dataset = pd.read_csv(DATA_DIRECTORY / "real_estate_dataset.csv")
    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return dataset.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).copy()


def encode_categories(dataset):
    """Fit a label encoder for each categorical feature."""
    encoders = {}
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        dataset[column] = encoder.fit_transform(dataset[column].astype(str))
        encoders[column] = encoder
    return encoders


def create_cnn(feature_count):
    """Build the compact 1D CNN used by the web application."""
    model = Sequential(
        [
            Input(shape=(feature_count, 1)),
            Conv1D(32, kernel_size=2, activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.15),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def save_metrics(test_prices, cnn_prices, forest_prices):
    """Store evaluation values for model-development reference."""
    metrics = {
        "cnn_r2": r2_score(test_prices, cnn_prices),
        "cnn_mae": mean_absolute_error(test_prices, cnn_prices),
        "rf_r2": r2_score(test_prices, forest_prices),
        "rf_mae": mean_absolute_error(test_prices, forest_prices),
    }
    (MODELS_DIRECTORY / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main():
    dataset = load_training_data()
    encoders = encode_categories(dataset)
    features = dataset[list(FEATURE_COLUMNS)]
    prices = dataset[TARGET_COLUMN]

    train_features, test_features, train_prices, test_prices = train_test_split(
        features, prices, test_size=0.20, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    MODELS_DIRECTORY.mkdir(exist_ok=True)
    joblib.dump(encoders, MODELS_DIRECTORY / "label_encoders.joblib")
    joblib.dump(scaler, MODELS_DIRECTORY / "scaler.joblib")

    forest = RandomForestRegressor(
        n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1
    )
    forest.fit(train_features, train_prices)
    forest_predictions = forest.predict(test_features)
    joblib.dump(forest, MODELS_DIRECTORY / "rf_model.joblib")

    train_cnn = train_scaled.reshape(-1, train_scaled.shape[1], 1)
    test_cnn = test_scaled.reshape(-1, test_scaled.shape[1], 1)
    cnn = create_cnn(train_cnn.shape[1])
    cnn.fit(train_cnn, train_prices, validation_split=0.15, epochs=60,
            batch_size=32, verbose=0)
    cnn_predictions = cnn.predict(test_cnn, verbose=0).flatten()
    cnn.save(MODELS_DIRECTORY / "cnn_model.h5")

    metrics = save_metrics(test_prices, cnn_predictions, forest_predictions)
    print("Training complete.")
    for metric, value in metrics.items():
        print(f"{metric}: {value:,.4f}")


if __name__ == "__main__":
    main()
