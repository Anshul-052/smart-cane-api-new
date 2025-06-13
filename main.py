from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import logging
import sys
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import easyocr
import os
import googlemaps
import re
from googletrans import Translator  # ✅ Added

# 📝 Logging Setup
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# 🌐 CORS Configuration
origins = [
    "https://6000-firebase-studio-1746954444859.cluster-fdkw7vjj7bgguspe3fbbc25tra.cloudworkstations.dev",
    "http://localhost:9002",
    "http://127.0.0.1:9002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Pydantic Models
class ImageData(BaseModel):
    image_base64: str

class ModeData(BaseModel):
    mode: str  # indoor, outdoor, road

class GPSRequest(BaseModel):
    destination: str

class LanguageRequest(BaseModel):
    language: str  # en, hi, mr, etc.

# 🌐 Globals
YOLO_MODEL_PATH = 'yolov8n.pt'
model = YOLO(YOLO_MODEL_PATH)
reader = easyocr.Reader(['en', 'hi'])
translator = Translator()  # ✅ Added

mode = "outdoor"
current_lang = "en"  # Default language

# 🌐 Supported Modes
mode_classes = {
    "indoor": [
        "person", "chair", "sofa", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush", "book", "cup", "fork", "knife", "spoon", "bowl", "bottle",
        "wine glass", "umbrella", "handbag", "tie", "suitcase", "pottedplant"
    ],
    "outdoor": [
        "person", "bicycle", "motorbike", "car", "bus", "train", "truck", "boat",
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "kite", "frisbee", "skateboard", "sports ball", "tennis racket",
        "baseball bat", "baseball glove", "surfboard", "snowboard", "skis"
    ],
    "road": [
        "person", "car", "motorbike", "bus", "truck", "bicycle", "train", "traffic light",
        "stop sign", "parking meter", "bench", "fire hydrant", "road sign"
    ]
}

# 🚗 Google Maps Setup
GOOGLE_MAPS_API_KEY = "AIzaSyAF37kDyerCWtnvCFxz8AqVsv_U_rRmBLI"
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

@app.get("/")
async def root():
    return {"message": "FastAPI server is running!"}

@app.post("/set_mode")
async def set_mode(data: ModeData):
    global mode
    if data.mode.lower() not in mode_classes:
        raise HTTPException(status_code=400, detail="Invalid mode. Use indoor, outdoor, or road.")
    mode = data.mode.lower()
    logger.info(f"Mode set to: {mode}")
    return {"message": f"Mode switched to {mode}"}

@app.post("/set_language")
async def set_language(data: LanguageRequest):
    global current_lang
    supported_languages = ['en', 'hi', 'mr', 'ta', 'te']
    if data.language not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    current_lang = data.language
    logger.info(f"Language switched to: {current_lang}")
    return {"message": f"Language set to {current_lang}"}

# 🔊 Speak Function
def speak(text, lang=None):
    global current_lang
    if lang is None:
        lang = current_lang
    try:
        if lang != 'en':
            translated = translator.translate(text, dest=lang)
            text = translated.text
            logger.debug(f"Translated Text ({lang}): {text}")

        tts = gTTS(text=text, lang=lang)
        tts.save("speech.mp3")
        sound = AudioSegment.from_mp3("speech.mp3")
        play(sound)
        os.remove("speech.mp3")
    except Exception as e:
        logger.error(f"Speech error: {e}")

# 📸 Base64 Decoder
def base64_to_cv2_image(data_uri):
    try:
        logger.debug(f"Received data URI (first 100 chars): {data_uri[:100]}")
        header, encoded = data_uri.split(",", 1)
        decoded_data = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error decoding base64: {e}")
        return None

@app.post("/start_detection")
async def start_detection(image_data: ImageData):
    try:
        cv2_img = base64_to_cv2_image(image_data.image_base64)
        if cv2_img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Object Detection
        results = model.predict(cv2_img, conf=0.5, device="cpu")
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            if label in mode_classes.get(mode, []):
                conf = float(box.conf[0].item())
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "box": [round(x1), round(y1), round(x2), round(y2)],
                })

        # OCR Text Recognition
        ocr_results = reader.readtext(cv2_img)
        detected_texts = []
        for (bbox, text, confidence) in ocr_results:
            if confidence > 0.7 and len(text.strip()) > 2 and text.isprintable():
                detected_texts.append(text.strip())

        logger.info(f"Detections: {detections}")
        logger.info(f"Detected Texts: {detected_texts}")

        # Speech Output
        sentence = ""
        if detections:
            object_list = ", ".join([d["label"] for d in detections])
            sentence += f"Detected objects: {object_list}. "
        if detected_texts:
            combined_text = " ".join(detected_texts)
            sentence += f"The text reads: {combined_text}."

        if sentence:
            speak(sentence)

        return {"detections": detections, "texts": detected_texts}

    except Exception as e:
        logger.error(f"Detection Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/start_gps_navigation")
async def start_gps_navigation(request: GPSRequest):
    try:
        origin = "Vishwakarma Institute of Technology, Kondhwa Campus"
        destination = request.destination
        directions_result = gmaps.directions(origin, destination, mode="walking")
        if not directions_result:
            raise HTTPException(status_code=404, detail="No route found")

        steps = directions_result[0]['legs'][0]['steps']
        directions_text = [re.sub('<[^<]+?>', '', step['html_instructions']) for step in steps]
        return {"instructions": directions_text}

    except Exception as e:
        logger.error(f"GPS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Navigation failed: {str(e)}")
