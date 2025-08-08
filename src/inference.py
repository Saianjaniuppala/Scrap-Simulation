# src/inference.py
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
import argparse
import os
from pathlib import Path

def load_pytorch_model(pth_path, device='cpu'):
    saved = torch.load(pth_path, map_location=device)
    from src.model import get_resnet18
    num_classes = len(saved['classes'])
    model = get_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    model.to(device)
    return model, saved['classes']

def preprocess_image(img_path):
    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return tf(img).unsqueeze(0)  # (1,C,H,W)

def predict_pytorch(model, classes, img_tensor, device='cpu'):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = np.argmax(probs)
        return classes[idx], float(probs[idx]), probs

def predict_onnx(onnx_path, img_tensor):
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    x = img_tensor.numpy().astype(np.float32)
    out = sess.run(None, {input_name: x})[0]
    probs = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)
    idx = np.argmax(probs[0])
    return idx, float(probs[0, idx]), probs[0]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='path to .pth (PyTorch) or .onnx')
    p.add_argument('--img', required=True)
    args = p.parse_args()
    ext = Path(args.model).suffix.lower()
    tensor = preprocess_image(args.img)
    if ext == '.pth':
        model, classes = load_pytorch_model(args.model)
        cls, conf, probs = predict_pytorch(model, classes, tensor)
        print({'class': cls, 'confidence': conf})
    elif ext == '.onnx':
        idx, conf, probs = predict_onnx(args.model, tensor)
        print({'class_idx': int(idx), 'confidence': conf})
    else:
        raise ValueError("Unsupported model format")
