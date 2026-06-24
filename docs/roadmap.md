# Roadmap and cloud GPU training runbook

## Build order (test-gated)

- M0 repo and tooling. Done.
- M1 scene and HAL. Done.
- M2 control (impedance, IK, primitives). Done.
- M3 scripted reseat demo. Done (working drive-reseat before any neural net).
- M4 domain randomization and data engine. Done (99% scripted seat success under DR).
- M5 LeRobot conversion and first train run. Built GPU-ready; full train deferred.
- M6 inference and eval. In progress / next.
- M7 custom hardware seam (b601 + off-the-shelf stubs).
- M8 dashboard and demo capture.

Stretch: residual RL / HIL-SERL to sharpen the contact policy, SmolVLA language
conditioning across bays, pi0.5 / GR00T N1.5 fine-tune, then the drive-swap task.

## Two-machine model

- Local box (Apple M1, CPU): fast iteration on the harness, control, data generation,
  conversion, and eval. Camera rendering uses the default CGL backend (no MUJOCO_GL).
- GPU box (A100 / 4090 class, Linux): set `MUJOCO_GL=egl`, render the dataset images at
  full resolution, and run `lerobot-train`. Never block local work on a training run.

## Environment

- Core (`robot_core`, `sim`, `data`, `eval`) is torch-free and runs on the system Python.
- The ML extras (`lerobot`, `torch`) install into a separate Python 3.11 venv
  (`.venv-train`) because lerobot v0.5 targets Python 3.12+ and torch arm64 wheels on
  3.13 are spotty. `scripts/setup.sh` creates it when Python 3.11 is available.

## Render the dataset (GPU box)

```bash
export MUJOCO_GL=egl
scripts/generate_demos.sh 500 datasets/rebot_scripted 224   # 224x224, ~17 fps
python -m data.dataset_validator datasets/rebot_scripted
```

Keep the large 224x224 set on the GPU box / Hub; keep a tiny 96-128px subset locally for
smoke tests. Local episodes are ~1.3 MB at 96x96, ~2.5 MB at 128x128.

## Convert to a LeRobotDataset

```bash
. .venv-train/bin/activate
python -m data.convert_to_lerobot --src datasets/rebot_scripted \
  --repo-id "$HF_USER/rebot_reseat" --root datasets/lerobot/rebot_reseat --verify
# optional: add --push to upload to the Hub
```

The converter is the only torch-touching file under `data/`; it maps `state` ->
`observation.state`, images -> `observation.images.{front,wrist}` (HWC uint8 in), action
-> `action`, and sets fps and per-frame task. Verify the v0.5/v3.0 API (`create`,
`add_frame` with a `task` key, `save_episode`, mandatory `finalize`) against the installed
wheel via `--help` and `inspect.signature` (the `[VERIFY]` markers).

## Train ACT (GPU box)

```bash
export HF_USER=...        # Hugging Face username
scripts/train_act.sh "$HF_USER/rebot_reseat" 100000
```

ACT trains in ~1-2 h on one A100 under 8 GB VRAM from tens to low hundreds of demos.
SmolVLA fine-tune is ~8 h. Diffusion is in between; pi0/pi0.5 are heavier VLAs. Monitor
with W&B; checkpoints land in `outputs/train/...`; push the policy with `--policy.repo_id`.

A 2-step CPU smoke of `lerobot-train` (in `tests/test_lerobot.py`, marked `lerobot`) loads
the converted dataset and runs two optimizer steps to catch schema/dtype/normalization
mismatches without a GPU. It auto-skips when the extra is absent.

## Eval loop

Pull a checkpoint to whichever box has the sim and run:

```bash
python -m eval.run_eval --controller scripted          # baseline, now
python -m eval.run_eval --controller policy --ckpt <path>   # learned, later
```

`lerobot-eval` is bound to its own gym envs, so reBot uses its own closed-loop runner
(`policies/infer.py` + `eval/run_eval.py`), with the scripted force-search controller as
the low-level seater.

## Dataset sync

Prefer the HF Hub as the source of truth (`push_to_hub` / pull on either box), or rsync the
v3.0 dataset directory. No secrets in the repo: `HF_USER` and tokens come from the env.
