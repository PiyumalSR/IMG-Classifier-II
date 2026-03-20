# Image Classifier API

A REST API that serves a MobileNetV2-based image classifier (10 classes) deployed on Render.com.

## Project Structure

```
├── app.py               # Flask API
├── model_weights.h5     # Trained Keras model weights
├── requirements.txt     # Python dependencies
├── render.yaml          # Render.com deployment config
├── Procfile             # Web server start command
└── README.md
```

## API Endpoints

### `GET /`
Health check — returns `{"status": "ok"}`.

### `POST /predict`
Upload an image and receive a prediction.

**Request:** `multipart/form-data` with a field named `file`.

**Response:**
```json
{
  "prediction": "cat",
  "confidence": 0.9231,
  "probabilities": {
    "airplane": 0.001,
    "cat": 0.9231,
    ...
  }
}
```

## Classes
The model predicts one of these 10 CIFAR-10 classes:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

> ⚠️ If you trained on a **different dataset**, update the `CLASS_NAMES` list in `app.py`.

---

## Local Testing

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py

# 3. Test with curl
curl -X POST http://localhost:5000/predict \
  -F "file=@your_image.jpg"
```

---

## Deploy to Render.com

### Step 1 — Prepare your GitHub repo
```bash
git init
git add app.py model_weights.h5 requirements.txt render.yaml Procfile .gitignore
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Create a Web Service on Render
1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo
4. Configure:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
5. Click **"Create Web Service"**

Render will automatically detect `render.yaml` if present.

### Step 3 — Test your live API
```bash
curl -X POST https://YOUR-APP.onrender.com/predict \
  -F "file=@your_image.jpg"
```

---

## ⚠️ Important Notes

- **RAM:** TensorFlow requires ~500MB RAM. Use at least Render's **Starter plan** ($7/mo).  
  The free tier (512MB) may crash on first load.
- **Cold starts:** Free/Starter instances sleep after inactivity. First request after sleep takes ~30 seconds.
- **`workers 1`:** Keep this at 1 — multiple workers would each load the model, causing out-of-memory errors.
- **Class names:** If your model was NOT trained on CIFAR-10, update `CLASS_NAMES` in `app.py`.
