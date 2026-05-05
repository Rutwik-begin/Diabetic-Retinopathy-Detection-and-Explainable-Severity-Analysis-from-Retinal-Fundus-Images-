from __future__ import annotations

import logging
from http import HTTPStatus

from flask import Flask, jsonify, request
from flask_cors import CORS

from explainability import generate_explainability_images
from inference import predict
from model_loader import load_model
from preprocessing import load_and_preprocess, pil_to_numpy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

MODEL, DEVICE, MODEL_PATH = load_model()
logger.info("Loaded model from %s on device %s", MODEL_PATH, DEVICE)


@app.get("/health")
def healthcheck():
    return jsonify(
        {
            "status": "ok",
            "model_path": str(MODEL_PATH),
            "device": str(DEVICE),
        }
    )


@app.post("/predict")
def predict_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file in form-data under key 'image'."}), HTTPStatus.BAD_REQUEST

    file_storage = request.files["image"]
    if not file_storage.filename:
        return jsonify({"error": "No file selected."}), HTTPStatus.BAD_REQUEST

    file_bytes = file_storage.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), HTTPStatus.BAD_REQUEST

    try:
        image, input_tensor = load_and_preprocess(file_bytes)
        prediction = predict(MODEL, input_tensor, DEVICE)
        explainability_images = generate_explainability_images(
            model=MODEL,
            input_tensor=input_tensor.to(DEVICE),
            original_rgb=pil_to_numpy(image),
            predicted_class=prediction.class_index,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), HTTPStatus.INTERNAL_SERVER_ERROR

    return jsonify(
        {
            "class_index": prediction.class_index,
            "class_name": prediction.class_name,
            "confidence": prediction.confidence,
            "medical_explanation": prediction.medical_explanation,
            "explainability_images": {
                "original.jpg": explainability_images["original.jpg"],
                "edges.jpg": explainability_images["edges.jpg"],
                "gradcam_overlay.jpg": explainability_images["gradcam_overlay.jpg"],
                "gradcam_heatmap.jpg": explainability_images["gradcam_heatmap.jpg"],
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
