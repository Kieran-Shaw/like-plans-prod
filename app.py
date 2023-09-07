import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from config.config import DevelopmentConfig, ProductionConfig
from methods.cold_start import ColdStart


# configure logging
def configure_logging(app):
    log_level = app.config.get("LOG_LEVEL", logging.INFO)

    if os.getenv("ENVIRONMENT") == "aws":
        # AWS environment, let CloudWatch handle logging.
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    else:
        # local or other environments, file and stream based logging.
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/application.log"),
            ],
        )


# create application
def create_app():
    app = Flask(__name__)
    app.logger = logging.getLogger(__name__)

    load_dotenv()  # load environment variables from .env file

    env = os.getenv("FLASK_ENV", "production")

    if env == "development":
        app.config.from_object(DevelopmentConfig)
    else:
        app.config.from_object(ProductionConfig)

    configure_logging(app)

    @app.route("/")
    def base_request():
        app.logger.info("Base URL")
        return jsonify(f"Base Endpoint, No Functionality"), 404

    @app.route("/cold-start", methods=["POST"])
    def cold_start():
        try:
            # parse input
            data = request.json
            # instantiate cold-start
            cold_start = ColdStart(data=data, logger=app.logger)
            # predict
            predicted_plans = cold_start.predict_plans()

            return predicted_plans

        except Exception as e:
            app.logger.error("Exception occurred", exc_info=True)
            return jsonify({"error": "An error occurred"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
