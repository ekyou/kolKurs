from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
import os
import io
import uuid
import json
import numpy as np
import traceback

app = FastAPI(title="Распознавание и детектирование автомобилей и людей на дороге (YOLOv8)")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = "best.pt"
classes_path = "classes.json"

model = None
classes = {}

if os.path.exists(model_path) and os.path.exists(classes_path):
    try:
        model = YOLO(model_path)

        with open(classes_path, 'r', encoding='utf-8') as f:
            classes = {int(k): v for k, v in json.load(f).items()}

        print("YOLO модель загружена успешно")
        print("Доступные классы:", classes)
        print("Количество классов:", len(classes))

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        model = None
        classes = {}
else:
    print("YOLO модель не найдена. Запустите train_yolov_8.py для обучения модели.")
    print(f"Ожидаемый путь к модели: {model_path}")
    print(f"Ожидаемый путь к классам: {classes_path}")
    model = None
    classes = {}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "classes": classes,
        "model_loaded": model is not None
    })


@app.post("/predict_image")
async def predict(request: Request, file: UploadFile = File(...)):
    if model is None:
        return templates.TemplateResponse("image_result.html", {
            "request": request,
            "error": "Модель не загружена. Сначала обучите модель."
        })

    try:
        if not file.content_type.startswith('image/'):
            return templates.TemplateResponse("image_result.html", {
                "request": request,
                "error": "Пожалуйста, загрузите файл изображения."
            })

        contents = await file.read()

        img = Image.open(io.BytesIO(contents)).convert('RGB')

        original_size = img.size

        target_size = (320, 320)

        img_resized = img.copy()
        img_resized.thumbnail(target_size, Image.Resampling.LANCZOS)

        new_img = Image.new('RGB', target_size, (255, 255, 255))

        img_resized_width, img_resized_height = img_resized.size
        left = (target_size[0] - img_resized_width) // 2
        top = (target_size[1] - img_resized_height) // 2
        new_img.paste(img_resized, (left, top))

        img_np = np.array(new_img)

        uploads_dir = "static/uploads/image"
        result_dir = "static/results/image"
        os.makedirs(uploads_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        input_path = f"{uploads_dir}/{uuid.uuid4().hex}_{file.filename}"
        result_path = f"{result_dir}/{uuid.uuid4().hex}_{file.filename}"

        img.save(input_path)

        results = model(
            img_np,
            conf=0.2,
            iou=0.25,
            max_det=1000,
            save=False
        )

        r = results[0]

        detections = []
        class_counts = {}
        total_count = 0

        scale_x = original_size[0] / target_size[0]
        scale_y = original_size[1] / target_size[1]

        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1_scaled = int((x1 - left) * scale_x)
                y1_scaled = int((y1 - top) * scale_y)
                x2_scaled = int((x2 - left) * scale_x)
                y2_scaled = int((y2 - top) * scale_y)

                x1_scaled = max(0, x1_scaled)
                y1_scaled = max(0, y1_scaled)
                x2_scaled = min(original_size[0], x2_scaled)
                y2_scaled = min(original_size[1], y2_scaled)

                class_name = classes.get(cls_id, str(cls_id))

                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_count += 1

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf * 100, 2),
                    "bbox": [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
                })

        annotated_img_original = img.copy()

        annotated_img_resized = r.plot()

        Image.fromarray(annotated_img_resized).save(result_path)

        formatted_counts = {}
        for class_name, count in class_counts.items():
            if count % 10 == 1 and count % 100 != 11:
                formatted_counts[class_name] = f"{count} обнаружен"
            elif count % 10 in [2, 3, 4] and count % 100 not in [12, 13, 14]:
                formatted_counts[class_name] = f"{count} обнаружено"
            else:
                formatted_counts[class_name] = f"{count} обнаружено"

        return templates.TemplateResponse("image_result.html", {
            "request": request,
            "image_url": f"/{result_path}",
            "class_counts": formatted_counts,
            "filename": file.filename,
            "total_count": total_count,
            "detections": detections,
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "processed_size": f"{target_size[0]}x{target_size[1]}"
        })

    except Exception as e:
        print(traceback.format_exc())

        return templates.TemplateResponse("image_result.html", {
            "request": request,
            "error": f"Ошибка обработки изображения: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)