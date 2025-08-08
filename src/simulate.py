# src/simulate.py
import time
import csv
import argparse
from pathlib import Path
from src.inference import preprocess_image, load_pytorch_model, predict_pytorch, predict_onnx
import random

def simulate_folder(folder, model_path, interval=0.5, out_csv='results/predictions.csv', threshold=0.6, use_onnx=False):
    folder = Path(folder)
    paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']])
    # prepare results
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','frame','predicted_class','pred_idx','confidence','low_confidence_flag','manual_override','notes'])
    # load model
    if use_onnx:
        onnx_path = model_path
    else:
        model, classes = load_pytorch_model(model_path)
    for i, p in enumerate(paths):
        start = time.time()
        img_tensor = preprocess_image(str(p))
        if use_onnx:
            idx, conf, probs = predict_onnx(onnx_path, img_tensor)
            pred_class = f"class_{idx}"
        else:
            pred_class, conf, probs = predict_pytorch(model, classes, img_tensor)
            idx = classes.index(pred_class)
        low_conf = conf < threshold
        # console output
        flag = "LOW_CONF" if low_conf else ""
        print(f"[{i+1}/{len(paths)}] {p.name} -> {pred_class} ({conf:.3f}) {flag}")
        # optional: manual override simulation (randomly simulate corrections)
        manual_override = False
        notes = ''
        # example: randomly mark 3% images as manually corrected (for active learning)
        if random.random() < 0.03:
            manual_override = True
            notes = "manually_flagged"
        # write CSV
        with open(out_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), p.name, pred_class, idx, f"{conf:.4f}", low_conf, manual_override, notes])
        # sleep to simulate frame rate
        elapsed = time.time() - start
        time.sleep(max(0, interval - elapsed))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--interval', type=float, default=0.3)
    ap.add_argument('--out', default='results/predictions.csv')
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--onnx', action='store_true')
    args = ap.parse_args()
    simulate_folder(args.folder, args.model, args.interval, args.out, args.threshold, use_onnx=args.onnx)
