from ultralytics import YOLO
import torch
import multiprocessing
import json

DATA_YAML = "dataset/data.yaml"
MODEL_NAME = "yolov8m.pt"
#MODEL_NAME = "runs/person_and_car/weights/last.pt"
EPOCHS = 30
IMGSZ = 320
BATCH = 12
DEVICE = 'cpu'
PROJECT = "runs/"
NAME = "person_and_car"
WORKERS = 0


def main():
    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        patience=30,
        cos_lr=True,
        plots=True,
        workers=WORKERS,
        #resume=True
    )

    print("\nОбучение!")
    print(f"Лучшая модель сохранена в: {PROJECT}/{NAME}/weights/best.pt")

    class_names = model.names

    with open(f"{PROJECT}/{NAME}/classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)

    print("Классы сохранены рядом с моделью")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
