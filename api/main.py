import os
import io
import uvicorn
import requests
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download



# Load the pre-trained model from huggingface_hub
model_path = hf_hub_download(
    repo_id = "alphagoal/Poultry_disease_prediction_model",
    filename = "poultry_disease_classifier.keras",
)

model = keras.models.load_model(model_path)

# Class names for the model predictions
class_labels = ["cocci","healthy","ncd","salmo"]

# Initialize FastAPI app
app = FastAPI(title="Poultry Disease Classifier API", version="1.0")

# Preprocess the image for model prediction
def preprocess_image(image_bytes):
    """Preprocess the image for model prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Resize to the input size of the model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict the class of a poultry disease from an image."""
    try:
        # Read and process the image
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis = 1)[0]
        predicted_class = class_labels[predicted_class_index]
        confidence = float(np.max(predictions))  # Get the confidence of the prediction
        # Return the prediction result
        return JSONResponse(
                            {"predicted_class": predicted_class,
                             "confidence": round(confidence,2)
                             })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)