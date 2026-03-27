import runpod
import sys
import traceback

sys.path.insert(0, '/pulid')

dit = None
t5 = None
clip_model = None
ae = None
pulid = None
INIT_ERROR = None


def load():
    global dit, t5, clip_model, ae, pulid

    import torch
    from flux.util import load_ae, load_clip, load_flow_model_quintized, load_t5
    from pulid.pipeline_flux import PuLIDPipeline

    print("[init] Flux FP8...", flush=True)
    dit = load_flow_model_quintized("flux-dev", "cpu")

    print("[init] T5...", flush=True)
    t5 = load_t5("cpu", max_length=128)

    print("[init] CLIP...", flush=True)
    clip_model = load_clip("cpu")

    print("[init] VAE (black-forest-labs/FLUX.1-dev, ~335MB)...", flush=True)
    ae = load_ae("flux-dev", "cpu")

    print("[init] PuLID...", flush=True)
    pulid = PuLIDPipeline(dit, "cpu", torch.bfloat16, onnx_provider="cpu")
    pulid.load_pretrain(version="v0.9.1")

    print("[init] Ready.", flush=True)


try:
    load()
except Exception:
    INIT_ERROR = traceback.format_exc()
    print(f"[init] FAILED:\n{INIT_ERROR}", flush=True)


def handler(job):
    if INIT_ERROR:
        return {"error": f"Init failed:\n{INIT_ERROR}"}

    import base64
    import io

    import numpy as np
    import torch
    from einops import rearrange
    from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
    from PIL import Image
    from pulid.utils import resize_numpy_image_long

    inp = job["input"]

    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    ref_b64 = inp.get("reference_face_base64")
    if not ref_b64:
        return {"error": "reference_face_base64 is required"}

    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))
    steps = int(inp.get("steps", 20))
    start_step = int(inp.get("start_step", 0))
    guidance = float(inp.get("guidance", 4.0))
    seed = int(inp.get("seed", -1))
    id_weight = float(inp.get("pulid_weight", 0.9))
    neg_prompt = inp.get("negative_prompt", "")
    true_cfg = float(inp.get("true_cfg", 1.0))
    timestep_to_start_cfg = int(inp.get("timestep_to_start_cfg", 1))
    max_seq_len = int(inp.get("max_sequence_length", 128))

    if seed == -1:
        seed = torch.Generator(device="cpu").seed()

    try:
        with torch.inference_mode():
            device = torch.device("cuda")

            t5.to(device)
            clip_model.to(device)

            x = get_noise(1, height, width, device=device, dtype=torch.bfloat16, seed=seed)
            pos_inp = prepare(t5, clip_model, x, prompt)

            neg_inp = None
            if true_cfg > 1.0:
                neg_inp = prepare(t5, clip_model, x, neg_prompt)

            t5.to("cpu")
            clip_model.to("cpu")
            torch.cuda.empty_cache()

            face_np = np.array(
                Image.open(io.BytesIO(base64.b64decode(ref_b64))).convert("RGB")
            )
            face_np = resize_numpy_image_long(face_np, 1024)

            pulid.components_to_device(device)
            id_emb, uncond_id_emb = pulid.get_id_embedding(
                face_np, cal_uncond=(true_cfg > 1.0)
            )
            pulid.components_to_device("cpu")
            torch.cuda.empty_cache()

            dit.to(device)
            timesteps = get_schedule(steps, pos_inp["img"].shape[1], shift=True)

            x = denoise(
                dit,
                pos_inp["img"],
                pos_inp["img_ids"],
                pos_inp["txt"],
                pos_inp["txt_ids"],
                pos_inp["vec"],
                timesteps,
                guidance=guidance,
                id=id_emb,
                id_weight=id_weight,
                start_step=start_step,
                uncond_id=uncond_id_emb,
                true_cfg=true_cfg,
                timestep_to_start_cfg=timestep_to_start_cfg,
                neg_txt=neg_inp["txt"] if neg_inp else None,
                neg_txt_ids=neg_inp["txt_ids"] if neg_inp else None,
                neg_vec=neg_inp["vec"] if neg_inp else None,
            )

            dit.to("cpu")
            torch.cuda.empty_cache()

            ae.to(device)
            x = unpack(x.float(), height, width)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = ae.decode(x)
            ae.to("cpu")
            torch.cuda.empty_cache()

            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        return {
            "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "format": "png",
            "seed": seed,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        t5.to("cpu")
        clip_model.to("cpu")
        dit.to("cpu")
        ae.to("cpu")
        pulid.components_to_device("cpu")
        torch.cuda.empty_cache()


runpod.serverless.start({"handler": handler})
