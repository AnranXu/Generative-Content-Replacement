## Generative-Content-Replacement
This is the interface for the trying generative content replacement (GCR), for CHI 2024 and SOUPS 2024 poster. 
For details of GCR implementation, please refers to our CHI 2024 paper "Examining Human Perception of Generative Content Replacement in Image Privacy Protection".
## Interface Layout
![Interface](https://github.com/AnranXu/Generative-Content-Replacement/assets/24409860/d4243f2c-0d1d-40e1-a5f0-2946c64e1fbe)

## Getting started
### 1. Prerequisite 
A GPU with >24Gb VRAM, 32Gb DRAM, 100Gb Storage
Nodejs, Anaconda
### 2. Installation
```bash
git clone https://github.com/AnranXu/Generative-Content-Replacement.git
```
```bash
npm install
```
```bash
cd Generative-Content-Replacement
conda create -n GCR python=3.10
conda activate GCR
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
cd backend
mkdir pretrained_models
cd pretrained_models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
### 3. Setting A Development Server To Try GCR
**Please do not deploy the code to any public servers. We do not ensure the security of the code.**
Turn on one bash for the backend
```bash
cd backend
python backend.py
```
Turn on another bash for the frontend
```bash
npm run start
```
Then, go to your broswer with the below address (do not forget to specify the ip of your PC or server that deploy GCR):
```bash
http://localhost:3000/?GCR_Server_IP=your_server_ip
```
