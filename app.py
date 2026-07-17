"""Application entry point for the Real Estate Price Predictor."""
from flask import Flask, jsonify, render_template, request

from services.data_repository import DataRepository
from services.model_service import ModelService, ValidationError
from utils.formatters import format_indian_currency


def create_app():
    """Create and configure the Flask application."""
    application = Flask(__name__)
    repository = DataRepository()
    prediction_service = ModelService()

    @application.get("/")
    def home():
        return render_template("index.html")

    @application.get("/predict_page")
    def prediction_page():
        return render_template(
            "predict.html",
            cities=repository.city_areas,
            property_types=repository.property_types,
            land_types=repository.land_types,
        )

    @application.get("/dashboard")
    def dashboard():
        return render_template(
            "dashboard.html",
            metrics=repository.dashboard_metrics(),
            cities=repository.city_areas,
        )

    @application.get("/transactions")
    def transactions():
        return render_template(
            "transactions.html",
            transactions=repository.recent_transactions(),
        )

    @application.get("/api/stats")
    def statistics():
        return jsonify(repository.statistics())

    @application.post("/predict")
    def predict_price():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(success=False, error="Send prediction details as JSON."), 400

        try:
            result = prediction_service.predict(payload)
        except ValidationError as error:
            return jsonify(success=False, error=str(error)), 400
        except Exception:
            application.logger.exception("Unexpected prediction error")
            return jsonify(success=False, error="The prediction could not be calculated."), 500

        currency, summary = format_indian_currency(result.price)
        return jsonify(
            success=True,
            predicted_price=currency,
            price_summary=summary,
            predicted_price_inr=round(result.price, 2),
            prediction_year=result.prediction_year,
            growth_rate=f"{result.growth_rate * 100:.1f}%",
            algorithm="1D CNN",
        )

    return application


app = create_app()
if __name__ == "__main__":
    app.run(debug=True, port=5000)
