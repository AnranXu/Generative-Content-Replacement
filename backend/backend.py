from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image
from model import DMBIS
import os 
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--device', type=str, default='cuda')
app = Flask(__name__)
CORS(app)
DMBIS = DMBIS(device=argparser.parse_args().device, mode='sd3')

def create_mask(vertices, image_width, image_height):
    # Create an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Convert vertices to a format that cv2.fillPoly expects
    pts = np.array([vertices], dtype=np.int32)  # Add brackets around vertices
    pts = pts.reshape((-1, 1, 2))  # Reshape pts to have a 3D shape

    # Fill the area defined by the vertices with 255
    cv2.fillPoly(mask, [pts], 255)  # Add brackets around pts

    return mask


@app.route('/api/create_mask', methods=['POST'])
def create_mask_route():
    data = request.get_json()
    x, y = data['point']['x'], data['point']['y']
    x, y = int(x), int(y)
    img = data['img']
    img = base64.b64decode(img.split(',')[1])
    img = Image.open(BytesIO(img))
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = np.array(img)
    masks, scores, logits = DMBIS.mask_predictor.predict_masks_with_sam(img, [[x, y]], [1])
    vertices = DMBIS.mask_predictor.output_polygon(masks, scores)
    masks = masks.astype(np.uint8) * 255
    best_mask = masks[np.argmax(scores)]

    # send image_filled and vertices to frontend
    send_file = {
        'mask': best_mask.tolist(),
        'vertices': vertices.tolist(),
    }
    return jsonify(send_file)

@app.route('/api/run_GCR', methods=['POST'])
def run_GCR_route():
    data = request.get_json()
    img = data['img']
    img = base64.b64decode(img.split(',')[1])
    img = Image.open(BytesIO(img))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = np.array(img)
    mask = data['mask']
    mask = np.array(mask)
    mask = mask.astype(np.uint8)
    prompt = data['prompt']
    strength = data['strength']
    strength = float(strength)
    text_strength = data['text_strength']
    text_strength = float(text_strength)
    img_filled, prompt = DMBIS.fill_with_stable_diffusion.fill_img_with_sd(img, mask, prompt, 
                                                                           strength=strength, guidance_scale=text_strength)
    # encode as base64
    # img_filled = Image.fromarray(img_filled)
    # buffered = BytesIO()
    # img_filled.save(buffered, format="JPEG")
    # img_filled = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # send image_filled and vertices to frontend
    send_file = {
        'GCRImage': img_filled.tolist(),
        'prompt': prompt,
    }
    return jsonify(send_file)

@app.route('/api/update_mask', methods=['POST'])
def update_mask_route():
    data = request.get_json()
    vertices = data['vertices']
    image_width = data['imageWidth']
    image_height = data['imageHeight']
        # Create the mask
    mask = create_mask(vertices, image_width, image_height)  # replace with actual image dimensions

    # Send the bytes back as a PNG image
    
    send_file = {
        'mask': mask.tolist(),
    }
    return jsonify(send_file)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
