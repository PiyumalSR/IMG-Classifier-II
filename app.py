import os
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import tensorflow as tf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Memory optimization ───────────────────────────────────────────────────────
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ── Build model architecture directly (avoids config.json compatibility) ──────
log.info("Building model architecture...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights=None
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(32, 32, 3))
x = tf.keras.layers.Resizing(96, 96)(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
log.info("Model built. Input shape: %s", model.input_shape)

# ── Load weights ──────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model.weights.h5")
log.info("Loading weights from: %s", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH)
log.info("✅ Weights loaded successfully.")

# ── Warm up ───────────────────────────────────────────────────────────────────
model(np.zeros((1, 32, 32, 3), dtype="float32"), training=False)
log.info("✅ Model warmed up and ready.")

# ── Class labels ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── CORS headers ──────────────────────────────────────────────────────────────
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    log.info("Health check hit.")
    return jsonify({"status": "ok", "message": "Image classifier API is running."})

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    log.info("POST /predict received.")

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    log.info("File received: %s", file.filename)

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        log.info("Image size: %s", img.size)

        img = img.resize((32, 32))
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model(img_array, training=False).numpy()
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        log.info("Prediction: %s (%.2f%%)", predicted_class, confidence * 100)

        probabilities = {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": probabilities
        })

    except Exception as e:
        log.error("Error: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Flask on port %d", port)
    app.run(host="0.0.0.0", port=port)
