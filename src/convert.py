# src/convert.py
import torch
import argparse
from pathlib import Path
from src.inference import load_pytorch_model
import torchvision

def to_torchscript(pth_path, out_path='models/resnet18_finetuned.pt', device='cpu'):
    model, classes = load_pytorch_model(pth_path, device=device)
    model.eval()
    example = torch.randn(1,3,224,224)
    traced = torch.jit.trace(model, example)
    traced.save(out_path)
    print("Saved TorchScript to", out_path)

def to_onnx(pth_path, out_path='models/resnet18_finetuned.onnx', device='cpu'):
    model, classes = load_pytorch_model(pth_path, device=device)
    model.eval()
    dummy = torch.randn(1,3,224,224, device=device)
    torch.onnx.export(model, dummy, out_path, opset_version=12,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    print("Saved ONNX to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--pth', required=True)
    p.add_argument('--onnx', default='models/resnet18_finetuned.onnx')
    p.add_argument('--pt', default='models/resnet18_finetuned.pt')
    args = p.parse_args()
    to_torchscript(args.pth, args.pt)
    to_onnx(args.pth, args.onnx)
