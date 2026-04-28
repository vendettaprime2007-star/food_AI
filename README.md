# Food Classifier ML Service 
 
Микросервис для классификации изображений еды. Использует MobileNetV3 (Transfer Learning) и FastAPI. 
 
## Запуск 
1. Установите зависимости: `pip install -r requirements.txt` 
2. Скачайте метки классов: скачать `imagenet_classes.txt` из репозитория PyTorch 
3. Запустите API: `uvicorn main:app --reload` 
 
## Gradio Demo 
Запустите `python gradio_app.py` для интерактивного интерфейса. 
