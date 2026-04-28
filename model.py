import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json

model = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1")
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with open('imagenet_classes.txt', 'r') as f:
    categories = [s.strip() for s in f.readlines()]

def predict_api(image_path_or_bytes):
    img = Image.open(image_path_or_bytes).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    results = []
    for i in range(top5_prob.size(0)):
        results.append({
            "class": categories[top5_catid[i]],
            "confidence": round(top5_prob[i].item(), 4)
        })
    return results