from PIL import Image
import io
import base64
import torch
from pinecone import Pinecone, ServerlessSpec
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import timm
import os
import cv2
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np

app = Flask(__name__)
CORS(app)

pc = Pinecone(api_key="pcsk_4VyhHx_GmqWehoqx3UzWdz1PDCoXddGaFGm79ernjVQtJd2CXYrcYbu5JgwVAzfAE2kp3y")
index = pc.Index(host="https://boulder-dino-giant-9nskykt.svc.aped-4627-b74a.pinecone.io")

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
model = AutoModel.from_pretrained('facebook/dinov2-giant')

MODEL_DIRECTORY = "model"

cfg = get_cfg()
cfg.merge_from_file(os.path.join(MODEL_DIRECTORY, "experiment_config.yml"))
cfg.MODEL.WEIGHTS = os.path.join(MODEL_DIRECTORY, "model_final.pth")
cfg.MODEL.DEVICE='cpu'

MetadataCatalog.get("meta").thing_classes = ["hold", "volume"]
metadata = MetadataCatalog.get("meta")

predictor = DefaultPredictor(cfg)

@app.route('/upsert', methods=['POST'])

def upsert():
    data = request.get_json(force=True)
    url = data.get('thumb_url')
    ID = data.get('ID')

    if not url:
        return jsonify({'error': 'No thumb_url'}), 400
    if not ID:
        return jsonify({'error': 'No ID'}), 400

    image = Image.open(requests.get(url, stream=True).raw)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    outputs = predictor(image)

    instances = outputs["instances"].to("cpu")
    CONF_THRESH = 0.8

    high_conf_mask = instances.scores >= CONF_THRESH
    instances = instances[high_conf_mask]

    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()  

        if masks.size == 0:
            print("No detections above threshold.")
        else:
            combined_mask = np.any(masks, axis=0)  
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            black_background = np.zeros_like(img_rgb)
            image = np.where(combined_mask[:, :, None], img_rgb, black_background)
    
    image = Image.fromarray(image)
    image = processor(images=image, return_tensors="pt")

    vector = model(**image)
    vector = vector.last_hidden_state.mean(dim=1) 
    vector = vector.squeeze().tolist()

    print('generated image')
    
    index.upsert(
        namespace="Default",
        vectors=[
            {
                "id": ID,
                "values": vector,
            },
        ],
    )

    return jsonify({'message': f'Upserted ID {ID}'})

@app.route('/save-thumb', methods=['POST'])
def save_thumb():
    
    data = request.get_json(force=True)
    print(f"Raw data: {data}")
    url = data.get('thumb_url')
    print(url)

    if not url:
        return jsonify({'error': 'No thumb_url'}), 400

    image = Image.open(requests.get(url, stream=True).raw)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    outputs = predictor(image)

    instances = outputs["instances"].to("cpu")
    CONF_THRESH = 0.8

    high_conf_mask = instances.scores >= CONF_THRESH
    instances = instances[high_conf_mask]

    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()  

        if masks.size == 0:
            print("No detections above threshold.")
        else:
            combined_mask = np.any(masks, axis=0)  
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            black_background = np.zeros_like(img_rgb)
            image = np.where(combined_mask[:, :, None], img_rgb, black_background)
    
    image = Image.fromarray(image)
    image = processor(images=image, return_tensors="pt")

    vector = model(**image)
    vector = vector.last_hidden_state.mean(dim=1)  
    vector = vector.squeeze().tolist()

    global latestVector
    latestVector = vector

    results = index.query(
        namespace="Default",
        vector=vector,
        top_k=4,
        metric='cosine'
    )

    matches = results.get('matches', [])
    if matches:
        response = {
            'id': matches[0]['id'], 
            'score': matches[0]['score'],
        }
    else:
        response = {'message': 'No matches found'}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
