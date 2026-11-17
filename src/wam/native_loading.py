"""Load a complete TrajWAM checkpoint for native waypoint generation."""
from pathlib import Path
import json
import struct


def checkpoint_files(directory):
    directory = Path(directory).expanduser().resolve()
    required = ["config.json", "generation_config.json", "preprocessor_config.json",
                "tokenizer_config.json", "tokenizer.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint: {', '.join(missing)}")
    if not ((directory / "chat_template.jinja").is_file()
            or json.loads((directory / "tokenizer_config.json").read_text()).get("chat_template")):
        raise FileNotFoundError("The checkpoint must include its chat template")
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if not weight_map:
            raise ValueError("Empty checkpoint index")
        files = sorted(set(weight_map.values()))
    elif (directory / "model.safetensors").is_file():
        files, weight_map = ["model.safetensors"], None
    else:
        raise FileNotFoundError("Supply all safetensors shards and their index")
    headers = {}
    for name in files:
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file():
            raise FileNotFoundError(f"Missing or invalid weight shard: {name}")
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if not 0 < length < 100_000_000:
                raise ValueError(f"Invalid safetensors header: {name}")
            header = json.loads(f.read(length))
        size = path.stat().st_size - 8 - length
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in headers or value["data_offsets"][1] > size:
                raise ValueError(f"Duplicate tensor or truncated shard: {key}")
            headers[key] = value
            if weight_map is not None and weight_map.get(key) != name:
                raise ValueError(f"Index disagrees with shard for {key}")
    if weight_map is not None and set(weight_map) != set(headers):
        raise ValueError("Checkpoint index and stored tensors disagree")
    return directory, headers


def load_native_model(model_path, device="auto"):
    import torch
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    directory, _ = checkpoint_files(model_path)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    if not getattr(config, "wam_enabled", False):
        raise ValueError("Use the trained TrajWAM checkpoint, not public initialization weights")
    # Resolve the constructor's DINO configuration without changing saved weights/configs.
    config.dinov3_config = str(Path(__file__).with_name("dinov3_config.json"))
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        directory, config=config, torch_dtype="auto", device_map=device,
        local_files_only=True, output_loading_info=True, ignore_mismatched_sizes=False,
        attn_implementation="sdpa",
    )
    problems = {key: info[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if problems:
        raise RuntimeError(f"Checkpoint is incompatible with the model: {problems}")
    if device == "auto" and {str(p.device) for p in model.parameters()} != {"cuda:0"}:
        raise RuntimeError("Insufficient GPU memory: native evaluation requires the full model on cuda:0")
    model.requires_grad_(False).eval()
    return processor, model, tokenizer
