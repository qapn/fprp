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

RUN pip install optimum-quanto==0.2.4 --no-deps --no-cache-dir

ENV HF_HOME=/models/hf_cache

RUN python -c "\
import torch, gc; \
from flux.util import load_flow_model_quintized, load_t5, load_clip; \
from pulid.pipeline_flux import PuLIDPipeline; \
print('[dl] Flux FP8 (XLabs-AI/flux-dev-fp8)', flush=True); \
dit = load_flow_model_quintized('flux-dev', 'cpu'); \
print('[dl] T5 (xlabs-ai/xflux_text_encoders)', flush=True); \
t = load_t5('cpu', 128); del t; gc.collect(); \
print('[dl] CLIP (openai/clip-vit-large-patch14)', flush=True); \
c = load_clip('cpu'); del c; gc.collect(); \
print('[dl] PuLID + InsightFace + EVA-CLIP', flush=True); \
p = PuLIDPipeline(dit, 'cpu', torch.bfloat16, onnx_provider='cpu'); \
p.load_pretrain(version='v0.9.1'); \
del p, dit; gc.collect(); \
print('[dl] Done', flush=True)"

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
