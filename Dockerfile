FROM madiator2011/better-pytorch:cuda12.4-torch2.6.0

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ToTheBeginning/PuLID.git /pulid
WORKDIR /pulid

RUN pip install \
    diffusers==0.30.0 transformers==4.43.3 \
    opencv-python-headless httpx==0.23.3 timm einops ftfy \
    facexlib insightface onnxruntime \
    accelerate sentencepiece safetensors torchsde \
    runpod --no-cache-dir

ENV HF_HOME=/models/hf_cache

RUN huggingface-cli download XLabs-AI/flux-dev-fp8 flux-dev-fp8.safetensors
RUN huggingface-cli download xlabs-ai/xflux_text_encoders
RUN huggingface-cli download openai/clip-vit-large-patch14
RUN huggingface-cli download guozinan/PuLID pulid_flux_v0.9.1.safetensors

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
