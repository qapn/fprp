FROM madiator2011/better-pytorch:cuda12.4-torch2.6.0

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ToTheBeginning/PuLID.git /pulid
WORKDIR /pulid

RUN pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

RUN pip install \
    diffusers==0.30.0 transformers==4.43.3 \
    opencv-python-headless httpx==0.23.3 timm einops ftfy \
    facexlib insightface onnxruntime \
    accelerate sentencepiece safetensors torchsde \
    runpod --no-cache-dir

ENV HF_HOME=/models/hf_cache

RUN huggingface-cli download Comfy-Org/flux1-dev flux1-dev-fp8.safetensors
RUN huggingface-cli download xlabs-ai/xflux_text_encoders
RUN huggingface-cli download openai/clip-vit-large-patch14
RUN huggingface-cli download guozinan/PuLID pulid_flux_v0.9.1.safetensors
RUN huggingface-cli download QuanSun/EVA-CLIP EVA02_CLIP_L_336_psz14_s6B.pt
RUN huggingface-cli download DIAMONIK7777/antelopev2 --local-dir /pulid/models/antelopev2

RUN mkdir -p /usr/local/lib/python3.10/dist-packages/facexlib/weights && \
    wget -q -O /usr/local/lib/python3.10/dist-packages/facexlib/weights/detection_Resnet50_Final.pth \
      https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -q -O /usr/local/lib/python3.10/dist-packages/facexlib/weights/parsing_parsenet.pth \
      https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth && \
    wget -q -O /usr/local/lib/python3.10/dist-packages/facexlib/weights/parsing_bisenet.pth \
      https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
