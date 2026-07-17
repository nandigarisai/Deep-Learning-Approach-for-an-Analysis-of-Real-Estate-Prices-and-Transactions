# Real Estate Price Prediction

A Flask-based academic project that estimates real-estate prices using a trained
1D CNN model and provides a view of recorded transactions across Andhra Pradesh.

## Project layout

- `app.py` — Flask application and HTTP routes
- `services/` — dataset access and model prediction logic
- `utils/` — display helpers, including Indian currency formatting
- `templates/` and `static/` — web interface assets
- `data/` — input datasets
- `models/` — saved model and preprocessing files
- `train_model.py` — script for retraining the models

## Run the project

Install the packages, then start the Flask server:

```powershell
py -m pip install -r requirements.txt
py app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Retrain the model

After updating `data/real_estate_dataset.csv`, run:

```powershell
py train_model.py
```

The script replaces the saved CNN model, Random Forest baseline, label encoders,
scaler, and development metrics in the `models/` directory.
