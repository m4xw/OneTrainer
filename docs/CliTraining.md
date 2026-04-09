# Training from CLI

All training functionality is available through the CLI command `python scripts/train.py`. The training configuration is
stored in a `.json` file that is passed to this script.

Some options require specifying paths to files with a specific
layout. These files can be created using the create_train_files.py script. You can call the script like
this `python scripts/create_train_files.py -h`.

To simplify the creation of the training config, you can export your settings from the UI by using the export button.
This will create a single file that contains every setting.

## Distillation Target Modes

Distillation is configured under `distillation` in your training config JSON.

### Required base fields

At minimum, set:

- `distillation.enabled`
- `distillation.parent_model_path`
- `distillation.parent_model_type`
- `distillation.target_mode`

Example:

```json
{
	"distillation": {
		"enabled": true,
		"parent_model_path": "models/parent.safetensors",
		"parent_model_type": "STABLE_DIFFUSION_XL_10_BASE",
		"target_mode": "RAW"
	}
}
```

### Target mode: `RAW`

- Legacy behavior.
- Uses parent prediction as distillation target without additional transformation.
- Safest default for compatibility.

```json
{
	"distillation": {
		"enabled": true,
		"target_mode": "RAW"
	}
}
```

### Target mode: `SCALED_LOSS_WEIGHT`

- **NEW**: Replaced legacy `CFG_SCALE` mode.
- Uses parent prediction directly; scaling applied only via `distillation.loss_weight` in loss calculation.
- No target transformation.
- Simpler than `CFG_SCALE` for pure knowledge distillation with adjustable strength.

Formula:

`prior_target = parent_prediction` (no transformation)

`distillation_loss *= distillation.loss_weight`

```json
{
	"distillation": {
		"enabled": true,
		"target_mode": "SCALED_LOSS_WEIGHT",
		"loss_weight": 1.0
	}
}
```

Practical notes:

- `loss_weight = 1.0` is standard; adjust to control distillation strength.
- Higher values increase the influence of parent model predictions.

### Target mode: `CFG_DISTILL`

- **NEW**: True classifier-free guidance distillation.
- Computes p_cfg target by blending empty (unconditional) and positive (conditional) parent predictions.
- Uses `cfg_scale` to control guidance strength.

Formula:

`p_cfg = parent_empty + cfg_scale * (parent_positive - parent_empty)`

```json
{
	"distillation": {
		"enabled": true,
		"target_mode": "CFG_DISTILL",
		"cfg_scale": 7.5
	}
}
```

Practical notes:

- Requires parent model to generate empty-prompt predictions (conditional on `target_mode`).
- `cfg_scale = 1.0` means no guidance; higher values strengthen guidance.
- **IMPORTANT**: Cache files generated with `CFG_DISTILL` include `predicted_empty`. Using `USE_CACHE` with a different `target_mode` will fail.

### Target mode: `CFG_UNDISTILLATION`

- Inverse CFG target transformation based on empty and positive parent predictions.
- Useful when you want the student to move toward parent-empty behavior while still staying in the standard loss pipeline.
- Uses `cfg_scale` to control inversion strength.

Formula:

`p_undistill = parent_empty - cfg_scale * (parent_positive - parent_empty)`

```json
{
	"distillation": {
		"enabled": true,
		"target_mode": "CFG_UNDISTILLATION",
		"cfg_scale": 1.0
	}
}
```

Practical notes:

- Currently supported on SDXL codepaths where `predicted_empty` is generated.
- Start with moderate values (for example 0.5-2.0) before increasing `cfg_scale`.
- Cache files generated with `CFG_UNDISTILLATION` also require `predicted_empty` and matching `target_mode` metadata.

### Target mode: `EMPTY_TARGET`

- Uses only the parent empty-prompt prediction as distillation target.
- This is the most direct "train student toward empty prediction" mode.

Formula:

`target = parent_empty`

```json
{
	"distillation": {
		"enabled": true,
		"target_mode": "EMPTY_TARGET"
	}
}
```

Practical notes:

- Currently supported on SDXL codepaths where `predicted_empty` is generated.
- `cfg_scale` is not used in this mode.
- Cache files generated with `EMPTY_TARGET` require `predicted_empty` and matching `target_mode` metadata.

### Target mode: `STEP_ROLLOUT`

- Generates a distilled target by blending multiple deterministic parent predictions.
- Useful as a step-distillation style target transform.
- Uses `rollout_steps` and `rollout_blend`.

```json
{
	"distillation": {
		"enabled": true,
		"target_mode": "STEP_ROLLOUT",
		"rollout_steps": 2,
		"rollout_blend": 0.5
	}
}
```

Practical notes:

- `rollout_steps` is clamped to at least `1`.
- `rollout_blend` is clamped to `[0.0, 1.0]`.
- `rollout_steps = 1` behaves like `RAW`.

### ⚠️ BREAKING CHANGE from `CFG_SCALE`

If you have existing configs using `target_mode: "CFG_SCALE"`, you **must** migrate:

**Old (no longer supported):**
```json
{
	"target_mode": "CFG_SCALE",
	"cfg_scale": 7.5
}
```

**New option 1 - Simple scaling (recommended if using base parent prediction):**
```json
{
	"target_mode": "SCALED_LOSS_WEIGHT",
	"loss_weight": 1.0
}
```

**New option 2 - CFG guidance:**
```json
{
	"target_mode": "CFG_DISTILL",
	"cfg_scale": 7.5
}
```

### Cache mode with target modes

When using cache mode, generated cache files are strategy-aware.

Recommended two-step workflow:

1. Run once with `distillation.cache_mode = "GENERATE_CACHE"`.
2. Train with `distillation.cache_mode = "USE_CACHE"`.

Important:

- Cache metadata now includes `target_mode`, `cfg_scale`, `rollout_steps`, `rollout_blend`, and (for empty-based modes) `predicted_empty`.
- If any of these differ between generation and usage, loading fails with a metadata mismatch error.
- In that case, regenerate cache with `GENERATE_CACHE`.

### Mixed batches and concept types

- Distillation loss is only applied to samples marked with concept type `DISTILLATION`.
- `PRIOR_PREDICTION` behavior remains separate.

### Complete example (CFG_DISTILL)

```json
{
	"distillation": {
		"enabled": true,
		"parent_model_path": "models/parent.safetensors",
		"parent_model_type": "STABLE_DIFFUSION_XL_10_BASE",
		"loss_type": "MSE",
		"loss_weight": 1.0,
		"kl_temperature": 1.0,
		"target_mode": "CFG_DISTILL",
		"cfg_scale": 7.5,
		"rollout_steps": 2,
		"rollout_blend": 0.5,
		"cache_mode": "DISABLED",
		"cache_dir": "workspace-cache/distillation"
	}
}
```
