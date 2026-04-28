from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import model

app = FastAPI(title="Food Classifier API")

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <h1>🍕 Классификатор еды</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*">
        <button type="submit">Распознать</button>
    </form>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {"error": "Загрузите изображение"}
    
    predictions = model.predict_api(file.file)
    return {"file_name": file.filename, "predictions": predictions}