from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
import joblib
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern
import tempfile
import os
import json

app = Flask(__name__)

# Load model and artifacts
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    model_info = joblib.load('models/model_info.pkl')
    print(f"✅ Model loaded: {model_info['algorithm']}")
    print(f"   Test Accuracy: {model_info['test_accuracy']:.4f}")
except:
    print("❌ Error loading model. Please train first.")
    model = None
    scaler = None
    model_info = {}

# ========================== HTML TEMPLATE ============================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arabic Digit Recognition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Arial, sans-serif; }
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;
               display: flex; justify-content: center; align-items: center; padding: 20px; }
        .container { background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                     width: 100%; max-width: 900px; overflow: hidden; }
        .header { background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; text-align: center; }
        .content { display: flex; min-height: 500px; }
        .left-panel, .right-panel { flex: 1; padding: 30px; }
        .left-panel { border-right: 2px solid #eee; }
        .drop-zone { border: 3px dashed #3498db; border-radius: 15px; padding: 40px; text-align: center; cursor: pointer;
                     transition: .3s; background: white; }
        .drop-zone.dragover, .drop-zone:hover { background: #f0f7ff; border-color: #2980b9; }
        .preview-container { display: none; margin-top: 20px; }
        .preview-img { max-width: 100%; max-height: 200px; border-radius: 10px;
                       box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .btn { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; border: none; padding: 15px 30px;
               font-size: 16px; font-weight: bold; border-radius: 10px; cursor: pointer; width: 100%;
               transition: .2s; margin-top: 20px; }
        .btn:disabled { background: #95a5a6; cursor: not-allowed; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px;
                   height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .result-container { display: none; }
        .result-container.show { display: block; animation: fadeIn .5s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .result-digit { font-size: 100px; font-weight: bold; text-align: center; color: #2c3e50; margin: 20px 0; }
        .arabic-digit { font-size: 120px; color: #e74c3c; }
        .confidence { background: #ecf0f1; padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center;
                      font-size: 18px; }
        .confidence-value { font-weight: bold; color: #27ae60; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔢 Arabic Digit Recognition</h1>
            <p>Upload an Arabic handwritten digit</p>
        </div>

        <div class="content">
            <div class="left-panel">
                <div class="drop-zone" id="dropZone">
                    <h3>Drag & Drop Image</h3>
                    <p id="fileName"></p>
                </div>
                <input type="file" id="fileInput" accept="image/*" hidden>

                <div class="preview-container" id="previewContainer">
                    <h3>Preview:</h3>
                    <img class="preview-img" id="previewImg">
                </div>

                <button class="btn" id="predictBtn" onclick="predict()" disabled>Recognize</button>

                <div class="loading" id="loading">
                    <div class="spinner"></div><p>Analyzing...</p>
                </div>
            </div>

            <div class="right-panel">
                <div class="result-container" id="resultContainer">
                    <h2>Result:</h2>
                    <div class="result-digit">
                        <span class="arabic-digit" id="arabicDigit">٠</span>
                        <div id="digitInfo">Digit: 0</div>
                    </div>

                    <div class="confidence">
                        Confidence: <span class="confidence-value" id="confidenceValue">0%</span>
                    </div>

                    <h3>Probabilities:</h3>
                    <div id="probabilities"></div>
                </div>
            </div>
        </div>
    </div>

<script>
fetch('/model_info')
    .then(r => r.json())
    .then(d => console.log("Model:", d));

let selectedFile = null;
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const previewContainer = document.getElementById('previewContainer');
const predictBtn = document.getElementById('predictBtn');
const resultContainer = document.getElementById('resultContainer');
const loading = document.getElementById('loading');
const fileName = document.getElementById('fileName');

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    predictBtn.disabled = false;

    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function predict() {
    if (!selectedFile) return;

    const fd = new FormData();
    fd.append('image', selectedFile);

    loading.style.display = 'block';
    resultContainer.classList.remove('show');
    predictBtn.disabled = true;

    fetch('/predict', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            loading.style.display = 'none';
            predictBtn.disabled = false;

            document.getElementById('arabicDigit').textContent =
                ['٠','١','٢','٣','٤','٥','٦','٧','٨','٩'][data.prediction];

            document.getElementById('digitInfo').textContent =
                `Digit: ${data.prediction} (${data.english_name})`;

            document.getElementById('confidenceValue').textContent =
                data.confidence.toFixed(1) + "%";

            const probsDiv = document.getElementById('probabilities');
            probsDiv.innerHTML = "";

            Object.entries(data.probabilities).forEach(([digit, prob]) => {
                probsDiv.innerHTML += `
                    <p>${digit}: ${prob.toFixed(1)}%</p>
                `;
            });

            resultContainer.classList.add('show');
        });
}
</script>

</body>
</html>
'''

# ====================== FEATURE EXTRACTION ============================
ARABIC_DIGITS = ["٠","١","٢","٣","٤","٥","٦","٧","٨","٩"]
ENGLISH_NAMES = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

def extract_features_from_image(img_array):
    features = []

    config = model_info.get("feature_config", {})

    if config.get("use_hog", True):
        hog_feat = hog(
            img_array,
            orientations=config.get("hog_orientations", 9),
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=None
        )
        features.extend(hog_feat)

    if config.get("use_lbp", True):
        lbp = local_binary_pattern(img_array, 24, 3, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        hist = hist.astype('float32') / (hist.sum() + 1e-6)
        features.extend(hist)

    return np.array(features)

# ======================= ROUTES ======================================

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/model_info')
def get_model_info():
    if model_info:
        return jsonify(model_info)
    return jsonify({"error": "Model not loaded"})

@app.route('/predict', methods=['POST'])
def predict_digit():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    request.files["image"].save(temp_file.name)

    try:
        img = Image.open(temp_file.name).convert("L")
        img_array = np.array(img)

        if np.mean(img_array) < 128:
            img = ImageOps.invert(img)
            img_array = np.array(img)

        img_size = model_info.get("feature_config", {}).get("img_size", (64, 64))
        img_array = cv2.resize(img_array, img_size)

        features = extract_features_from_image(img_array)
        features_scaled = scaler.transform([features])

        pred = model.predict(features_scaled)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features_scaled)[0]
        else:
            prob = np.zeros(10); prob[pred] = 1

        prob_dict = {str(i): float(prob[i] * 100) for i in range(10)}

        return jsonify({
            "prediction": int(pred),
            "arabic_digit": ARABIC_DIGITS[pred],
            "english_name": ENGLISH_NAMES[pred],
            "confidence": float(prob[pred] * 100),
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(temp_file.name)

@app.route('/test_sample')
def test_sample():
    samples = []

    for i in range(10):
        path = f"data/{i}/sample.png"
        if not os.path.exists(path):
            continue

        try:
            img = Image.open(path).convert("L")
            img_array = np.array(img)

            if np.mean(img_array) < 128:
                img = ImageOps.invert(img)
                img_array = np.array(img)

            img_size = model_info.get("feature_config", {}).get("img_size", (64, 64))
            img_array = cv2.resize(img_array, img_size)

            features = extract_features_from_image(img_array)
            scaled = scaler.transform([features])

            pred = model.predict(scaled)[0]

            samples.append({
                "digit": i,
                "predicted": int(pred),
                "correct": bool(i == pred),
                "arabic_digit": ARABIC_DIGITS[pred]
            })

        except:
            pass

    return jsonify(samples)

# ======================= RUN APP ===============================
if __name__ == "__main__":
    app.run(debug=True)
