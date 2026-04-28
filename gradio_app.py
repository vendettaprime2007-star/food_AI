import gradio as gr
from model import predict_api

def classify_image(img):
    predictions = predict_api(img)
    return {p['class']: p['confidence'] for p in predictions}

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=5),
    title="Food Image Classifier"
)

iface.launch()