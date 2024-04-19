## Generative-Content-Replacement
This is the interface for the generative content replacement (GCR), for SOUPS 2024 poster. 
For details of GCR, CHI 2024 paper "Examining Human Perception of Generative Content Replacement in Image Privacy Protection".

## Getting started
### 1. Prerequisite 
A GPU with >24Gb VRAM, 32Gb DRAM
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
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
### 3. Setting A Development Server To Try GCR
**Please do not deploy the code to any public servers. We do not ensure the security of the code.**
