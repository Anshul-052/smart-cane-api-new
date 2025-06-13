import cv2
import numpy as np
import easyocr
import threading
import time
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en', 'hi'])  # Add more as needed

cap = cv2.VideoCapture(0)
prev_time = 0
speech_interval = 3  # seconds
speech_thread = None

def speak(text, lang='en'):
    global speech_thread

    def run():
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save("speech.mp3")
            sound = AudioSegment.from_mp3("speech.mp3")
            play(sound)
            os.remove("speech.mp3")
        except Exception as e:
            print(f"Speech Error: {e}")

    # Stop previous thread if running
    if speech_thread and speech_thread.is_alive():
        return  # Skip if last one is still speaking

    speech_thread = threading.Thread(target=run)
    speech_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    detected_objects = set()
    detected_text = ""

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box)
                label = model.names[class_id]
                detected_objects.add(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                roi = frame[y1:y2, x1:x2]
                results_ocr = reader.readtext(roi)
                for result in results_ocr:
                    text = result[1].strip()
                    if text:
                        detected_text += text + " "

    current_time = time.time()
    if current_time - prev_time > speech_interval:
        sentence = ""
        if detected_objects:
            sentence += f"I see {', '.join(detected_objects)}. "
        if detected_text:
            sentence += f"The text reads: {detected_text}"

        if sentence:
            lang = 'hi' if any('\u0900' <= ch <= '\u097F' for ch in sentence) else 'en'
            speak(sentence, lang)
            prev_time = current_time

    cv2.imshow("Smart Cane Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
