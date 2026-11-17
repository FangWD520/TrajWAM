#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CARLA_ROOT:?Set CARLA_ROOT to the CARLA installation}"
: "${TEAM_CONFIG:?Set TEAM_CONFIG to the complete TrajWAM checkpoint}"
: "${SAVE_PATH:?Set SAVE_PATH to the output directory for this worker}"
export CARLA_ROOT="$(realpath "$CARLA_ROOT")"
export TEAM_CONFIG="$(realpath "$TEAM_CONFIG")"
export SAVE_PATH="$(realpath -m "$SAVE_PATH")"
export GPU_RANK="${GPU_RANK:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_RANK"
export PYTHONPATH="$PWD/src:$PWD/bench2drive:$PWD/bench2drive/leaderboard:$PWD/bench2drive/scenario_runner:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
export CARLA_SERVER="$CARLA_ROOT/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT="$PWD/bench2drive/scenario_runner"
export LEADERBOARD_ROOT="$PWD/bench2drive/leaderboard"
export IS_BENCH2DRIVE=True PLANNER_TYPE=only_traj
export ROUTES="${ROUTES:-$PWD/bench2drive/leaderboard/data/bench2drive220.xml}"
python scripts/check_environment.py --checkpoint "$TEAM_CONFIG"
mkdir -p "$SAVE_PATH"
cd bench2drive
exec python leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes="$ROUTES" --agent=team_code/qwen_b2d_agent.py --agent-config="$TEAM_CONFIG" \
  --checkpoint="${CHECKPOINT_ENDPOINT:-$SAVE_PATH/results.json}" \
  --debug-checkpoint="$SAVE_PATH/live_results.txt" \
  --port="${PORT:-2000}" --traffic-manager-port="${TM_PORT:-8000}" \
  --traffic-manager-seed="${TM_SEED:-0}" --timeout="${CLIENT_TIMEOUT:-100}" \
  --gpu-rank="$GPU_RANK" --repetitions=1 --track=SENSORS --resume="${RESUME:-True}"
