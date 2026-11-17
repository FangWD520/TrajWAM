# Closed-loop reproduction

Run the commands from the repository root (`TrajWAM/`).

## 1. Environment and simulator

Use Linux, an NVIDIA CUDA GPU, and a dedicated Python 3.10 or 3.11 environment.
Each evaluation worker runs one planner and one CARLA instance on its GPU.
Allocate sufficient GPU memory for both processes and disk space for saved
camera frames; the agent reads these frames during inference.

The supplied evaluation runtime uses CARLA 0.9.16. Install its server, matching
Python API and additional town assets using the
[official CARLA installation instructions](https://carla.readthedocs.io/en/latest/start_quickstart/)
and the [Bench2Drive setup instructions](https://github.com/Thinklab-SJTU/Bench2Drive).
Set `CARLA_ROOT` to the extracted server directory containing `CarlaUE4.sh` and
`PythonAPI/`. Install all towns used by `bench2drive220.xml`, including the large maps.

```bash
conda create -n trajwam-eval python=3.11 -y
conda activate trajwam-eval
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install carla==0.9.16
bash scripts/setup.sh
export CARLA_ROOT=/absolute/path/to/CARLA_0.9.16
export PYTHONPATH="$PWD/src:$PWD/bench2drive:$PWD/bench2drive/leaderboard:$PWD/bench2drive/scenario_runner:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
```

Use a driver compatible with CUDA 12.8. The setup script installs Transformers
4.56.0 and the supplied Qwen2.5-VL/DINOv3 model patches into this environment.
FlashAttention is not required; the loader uses SDPA.

CARLA runs under a non-root Linux account. Run evaluation as that account.
On a root-only rental instance, create a non-root account with access to
`CARLA_ROOT` and set `CARLA_USER` to its name; the launcher starts only CARLA
under that account. Ensure the account has GPU-device and directory access.

Keep the bundled route XML, scenario code and scoring rules together. Do not
replace them with a different Bench2Drive release. Closed-loop evaluation
generates observations in CARLA and does not require downloading the training
dataset. Dataset resources are available through the
[official Bench2Drive repository](https://github.com/Thinklab-SJTU/Bench2Drive#dataset).

## 2. Prepare the trained checkpoint

Use the complete TrajWAM checkpoint associated with the main closed-loop result.
Download and verification instructions are in [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md).
Weights are distributed separately from the code. Public DeepSight weights are an
initialization checkpoint and do not reproduce the trained TrajWAM result.

Place the checkpoint in a local directory containing:

```
checkpoints/trajwam/
  config.json
  generation_config.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  tokenizer.json
  tokenizer_config.json
  preprocessor_config.json
  chat_template.jinja
  ... remaining tokenizer/processor files from the same checkpoint
```

Keep all tokenizer, special-token, vocabulary and processor files from that
checkpoint. Optimizer states, training logs and trainer state are unnecessary.
The loader resolves the DINO constructor configuration from this package in
memory and leaves the saved checkpoint unchanged. It rejects missing or
unexpected model tensors rather than initializing them randomly.

```bash
python scripts/check_environment.py \
  --checkpoint /absolute/path/to/checkpoints/trajwam --files-only
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py \
  --checkpoint /absolute/path/to/checkpoints/trajwam --load-model
```

## 3. Evaluate all 220 routes

For four GPUs, the launcher partitions the routes into four disjoint shards of
55 routes, with independent RPC/traffic-manager ports, outputs and logs.

```bash
python scripts/evaluate_multigpu.py \
  --gpus 0,1,2,3 \
  --checkpoint /absolute/path/to/checkpoints/trajwam \
  --output /absolute/path/to/results/trajwam220 \
  --dry-run

python scripts/evaluate_multigpu.py \
  --gpus 0,1,2,3 \
  --checkpoint /absolute/path/to/checkpoints/trajwam \
  --output /absolute/path/to/results/trajwam220
```

Use `--gpus 0` for a single GPU; it evaluates the same 220 routes serially.
Use `--port` and `--tm-port` if the default port ranges are occupied. The default
traffic-manager seed is 0, inherited from the bundled evaluator; set `--tm-seed`
explicitly when matching a run that used a different simulator seed.
The planner retains the checkpoint generation settings and its native
`max_new_tokens=15000` call. Sensors, history preprocessing and PID are unchanged.

Each output directory contains:

```
evaluation.json       Checkpoint path and launch settings
shards.json           GPU assignment and route IDs
routes_*.xml          Per-worker routes
result_shard*.json    Benchmark route records
log_shard*.txt        Worker logs
frames_*/             Camera frames and per-route control metrics
merged.json           All 220 route records and aggregate scores
summary.json          DS and SR (%)
```

To resume an interrupted evaluation, repeat the same command with `--resume`.
Keep the checkpoint, GPU/shard allocation and output directory unchanged.
The benchmark resume mechanism retains completed routes, including failed
driving episodes. Inspect worker logs after infrastructure failures.

## 4. Compute DS and SR

The launcher aggregates after all workers finish. To run aggregation separately:

```bash
python scripts/aggregate_results.py \
  --output /absolute/path/to/results/trajwam220
```

Aggregation requires exactly the 220 expected unique route IDs. DS is the mean
of `scores.score_composed` over those routes. Strict success requires status
`Completed` or `Perfect` and no nonempty infraction list except
`min_speed_infractions`. SR is 100 times the number of strict successes divided
by 220. `summary.json` stores SR in percent; `merged.json` retains the official
0–1 success-rate convention. Failed driving routes remain in the denominator.

## 5. Ability, efficiency and smoothness

After the closed-loop workers stop, run the bundled ability tool. It starts
its own CARLA server to resolve junctions used by the traffic-sign metric:

```bash
python bench2drive/tools/ability_benchmark.py \
  -f bench2drive/leaderboard/data/bench2drive220.xml \
  -r /absolute/path/to/results/trajwam220/merged.json \
  --output /absolute/path/to/results/trajwam220/ability.json \
  --gpu 0 --port 60100
```

For efficiency and smoothness, collect the generated `metric_info.json` files
into the route-name layout expected by the benchmark, then run its scorer:

```bash
python scripts/prepare_route_metrics.py \
  --output /absolute/path/to/results/trajwam220
python bench2drive/tools/efficiency_smoothness_benchmark.py \
  -f /absolute/path/to/results/trajwam220/merged.json \
  -m /absolute/path/to/results/trajwam220/route_metrics
```
