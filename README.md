# TrajWAM

**Trajectory-Conditioned World Action Modeling for Vision–Language Driving**

Xuerui Fang, Gulimila Kezierbieke, CHEN WANGXING, Hui Li

Xinjiang Agricultural University · ToReality (Hangzhou) · Jiangnan University

[Evaluation guide](docs/REPRODUCIBILITY.md) · [Model download](docs/MODEL_DOWNLOAD.md) · [Model card](docs/MODEL_CARD.md)

TrajWAM trains a vision–language driving policy with trajectory-conditioned BEV
future learning. Factual reconstruction and query separation provide auxiliary
training signals to the shared planner. Native inference generates waypoints
directly and uses PID control.

| Evaluation | Driving Score ↑ | Success Rate (%) ↑ |
| --- | ---: | ---: |
| TrajWAM — all 220 Bench2Drive routes | 91.53 | 75.45 |

This release contains the closed-loop evaluation runtime: the model loader,
driving agent, benchmark scenarios, single/multiple GPU launcher, and scoring
tools. Follow the [evaluation guide](docs/REPRODUCIBILITY.md) to install the
environment, prepare the complete checkpoint, and evaluate all 220 routes.

## Quick start

```bash
git clone https://github.com/FangWD520/TrajWAM.git
cd TrajWAM
# Complete environment, simulator and checkpoint setup in docs/REPRODUCIBILITY.md.
python scripts/evaluate_multigpu.py \
  --gpus 0,1,2,3 \
  --checkpoint /absolute/path/to/TrajWAM-v1.0.0 \
  --output /absolute/path/to/results/trajwam220
```

Use `--gpus 0` to run the same routes on a single GPU. Training data is available
through the [official Bench2Drive repository](https://github.com/Thinklab-SJTU/Bench2Drive#dataset);
closed-loop evaluation obtains its observations from CARLA.

## Repository layout

```
assets/              Historical-image placeholder
bench2drive/         Route XML, scenarios, evaluator, agent, PID and scoring
scripts/             Installation, checkpoint checks, launch and DS/SR aggregation
src/wam/             Checkpoint-compatible modules and native loading
src/transformers/    Version-locked model runtime patch
```

The runtime preserves the checkpoint's auxiliary module tensors for complete
loading; native generation uses the VLM and language-model head. Training and
auxiliary prediction entry points are excluded from this evaluation package.

## Citation

```bibtex
@misc{fang2026trajwam,
  title  = {TrajWAM: Trajectory-Conditioned World Action Modeling for Vision--Language Driving},
  author = {Fang, Xuerui and Kezierbieke, Gulimila and {CHEN WANGXING} and Li, Hui},
  year   = {2026},
  url    = {https://github.com/FangWD520/TrajWAM}
}
```

## Acknowledgments and licenses

Built with Qwen. This code builds on DeepSight, Transformers, Bench2Drive and
CARLA. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream
attribution and code licenses. The checkpoint is distributed under the
applicable [Qwen Research License](licenses/QWEN_RESEARCH_LICENSE.txt) and
[DINOv3 License](licenses/DINOV3_LICENSE.md); the code license does not replace
these model terms.

## Contact

Corresponding authors: Gulimila Kezierbieke (`glml@xjau.edu.cn`) and
Hui Li (`lihui.cv@jiangnan.edu.cn`).
