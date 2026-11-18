# TrajWAM checkpoint

Built with Qwen.

## Model and task

TrajWAM is a vision–language planner for autonomous-driving research in CARLA.
It consumes historical front-camera images, current surround-camera images,
ego state and route instructions, and generates reasoning and driving
waypoints. The supplied agent applies PID control to those waypoints.

The complete checkpoint contains 4,085,644,803 parameters, including the
Qwen2.5-VL-3B-based planner and auxiliary modules. Model tensors are stored in
BF16 across two safetensors shards. Native waypoint generation bypasses the
auxiliary future-prediction branch; the loader retains its tensors for full
checkpoint compatibility.

## Evaluation

The paper reports Driving Score 91.53 and strict Success Rate 75.45% on all
220 Bench2Drive routes. The release includes the route XML, scenario code,
agent, controller and score aggregation used by the evaluation runtime.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the simulator version and
execution settings. This release supports simulator research and evaluation;
it has not been validated for deployment on public roads.

## Files and provenance

The release uses the selected `wam-stage1-v6/checkpoint-3542` model. The supplied
runtime loads its safetensors files unchanged and resolves the DINO constructor
configuration using the configuration bundled with the code. Tokenizer,
processor and generation settings come from the checkpoint. Optimizer states
and training logs are unnecessary for inference. Model-shard checksums are
provided in [WEIGHT_SHARDS_SHA256SUMS](WEIGHT_SHARDS_SHA256SUMS).

## Licenses

The Qwen-derived weights use the [Qwen Research License](../licenses/QWEN_RESEARCH_LICENSE.txt).
The included DINOv3 component uses the [DINOv3 License](../licenses/DINOV3_LICENSE.md).
Their terms and [NOTICE](../NOTICE) are included in this repository. Upstream code
attribution appears in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
