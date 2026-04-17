import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


class DistillationCacheManager:
    """
    Manages caching of parent model predictions for distillation training.
    
    This allows a two-step distillation workflow:
    1. Generate cache: Run parent model inference and save predictions to disk
    2. Use cache: Load cached predictions instead of running parent model
    
    Benefits:
    - Reduces VRAM usage during training (no need to load parent model)
    - Faster training when using same dataset for multiple epochs
    - Avoids model swapping overhead on low-VRAM systems
    """
    
    def __init__(
        self,
        cache_dir: str,
        parent_model_path: str,
        parent_model_type: str,
        target_mode: str,
        cfg_scale: float,
        rollout_steps: int,
        rollout_blend: float,
    ):
        """
        Initialize the distillation cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            parent_model_path: Path to parent model (for validation)
            parent_model_type: Type of parent model (for validation)
        """
        self.cache_dir = Path(cache_dir)
        self.parent_model_path = parent_model_path
        self.parent_model_type = parent_model_type
        self.target_mode = target_mode
        self.cfg_scale = cfg_scale
        self.rollout_steps = rollout_steps
        self.rollout_blend = rollout_blend
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DistillationCacheManager] Initialized cache directory: {self.cache_dir.absolute()}")
        print(f"[DistillationCacheManager] Directory exists: {self.cache_dir.exists()}")
        print(f"[DistillationCacheManager] Directory writable: {os.access(self.cache_dir, os.W_OK)}")
        
        # Metadata file for cache validation
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _generate_cache_key(self, image_path: str, timestep: int) -> str:
        """
        Generate a unique cache key from image path and timestep.
        
        Uses hash to avoid filesystem path issues (special chars, length limits).
        
        Args:
            image_path: Path to the training image
            timestep: The timestep used for this prediction
            
        Returns:
            Hash-based cache key
        """
        # Combine image path and timestep for uniqueness
        key_string = f"{image_path}_{timestep}"
        # Use SHA256 hash (first 16 chars should be sufficient for uniqueness)
        hash_object = hashlib.sha256(key_string.encode())
        return hash_object.hexdigest()[:16]
    
    def _get_cache_filepath(self, cache_key: str) -> Path:
        """
        Get the full filepath for a cache entry.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pt"
    
    def save_prediction(
        self,
        image_path: str,
        timestep: int,
        prediction_dict: dict[str, Tensor],
        global_step: int = 0,
    ) -> None:
        """
        Save a parent model prediction to cache.
        
        Args:
            image_path: Path to the training image
            timestep: The timestep used for this prediction
            prediction_dict: Dictionary containing prediction tensors
                            Should include: 'predicted', 'target', 'prediction_type'
            global_step: Current training step (for debugging)
        """
        cache_key = self._generate_cache_key(image_path, timestep)
        cache_file = self._get_cache_filepath(cache_key)
        
        # Prepare cache data
        cache_data = {
            'predicted': prediction_dict.get('predicted').cpu() if prediction_dict.get('predicted') is not None else None,
            'target': prediction_dict.get('target').cpu() if prediction_dict.get('target') is not None else None,
            'predicted_empty': prediction_dict.get('predicted_empty').cpu() if prediction_dict.get('predicted_empty') is not None else None,
            'prediction_type': prediction_dict.get('prediction_type'),
            'timestep': timestep,
            'image_path': image_path,
            'global_step': global_step,
            'parent_model_path': self.parent_model_path,
            'parent_model_type': self.parent_model_type,
            'target_mode': self.target_mode,
            'cfg_scale': self.cfg_scale,
            'rollout_steps': self.rollout_steps,
            'rollout_blend': self.rollout_blend,
        }
        
        # CFG_REGULARISE metadata (embedded when available)
        if 'cfg_regularise_initial_loss' in prediction_dict:
            cache_data['cfg_regularise_initial_loss'] = prediction_dict['cfg_regularise_initial_loss']
        if 'cfg_regularise_image_path' in prediction_dict:
            cache_data['cfg_regularise_image_path'] = prediction_dict['cfg_regularise_image_path']
        
        try:
            # Save to disk
            torch.save(cache_data, cache_file)
        except Exception as e:
            print(f"[DistillationCacheManager] ERROR saving cache file {cache_file}: {str(e)}")
            raise
    
    def load_prediction(self, image_path: str, timestep: int) -> dict[str, Any] | None:
        """
        Load a cached parent model prediction.
        
        Args:
            image_path: Path to the training image
            timestep: The timestep to load
            
        Returns:
            Dictionary with cached prediction data, or None if not found
        """
        cache_key = self._generate_cache_key(image_path, timestep)
        cache_file = self._get_cache_filepath(cache_key)
        
        if not cache_file.exists():
            self.cache_misses += 1
            return None
        
        try:
            cache_data = torch.load(cache_file, map_location='cpu')
            
            # Validate metadata
            if cache_data.get('parent_model_path') != self.parent_model_path:
                raise ValueError(
                    f"Cache metadata mismatch: parent_model_path\n"
                    f"  Expected: {self.parent_model_path}\n"
                    f"  Got: {cache_data.get('parent_model_path')}"
                )
            
            if cache_data.get('parent_model_type') != self.parent_model_type:
                raise ValueError(
                    f"Cache metadata mismatch: parent_model_type\n"
                    f"  Expected: {self.parent_model_type}\n"
                    f"  Got: {cache_data.get('parent_model_type')}"
                )

            #if cache_data.get('target_mode') != self.target_mode:
            #    raise ValueError(
            #        f"Cache metadata mismatch: target_mode\n"
            #        f"  Expected: {self.target_mode}\n"
            #        f"  Got: {cache_data.get('target_mode')}"
            #    )

            #if float(cache_data.get('cfg_scale', 1.0)) != float(self.cfg_scale):
            #    raise ValueError(
            #        f"Cache metadata mismatch: cfg_scale\n"
            #        f"  Expected: {self.cfg_scale}\n"
            #        f"  Got: {cache_data.get('cfg_scale')}"
            #    )

            if int(cache_data.get('rollout_steps', 1)) != int(self.rollout_steps):
                raise ValueError(
                    f"Cache metadata mismatch: rollout_steps\n"
                    f"  Expected: {self.rollout_steps}\n"
                    f"  Got: {cache_data.get('rollout_steps')}"
                )

            if float(cache_data.get('rollout_blend', 0.5)) != float(self.rollout_blend):
                raise ValueError(
                    f"Cache metadata mismatch: rollout_blend\n"
                    f"  Expected: {self.rollout_blend}\n"
                    f"  Got: {cache_data.get('rollout_blend')}"
                )
            
            # Validate predicted_empty for CFG-based target modes
            if self.target_mode in {'CFG_DISTILL', 'EMPTY_TARGET', 'CFG_REGULARISE'}:
                if cache_data.get('predicted_empty') is None:
                    raise ValueError(
                        f"Cache data for {self.target_mode} mode missing predicted_empty.\n"
                        f"This cache was likely generated with a different target_mode.\n"
                        f"Please regenerate the cache with matching target_mode."
                    )
            
            self.cache_hits += 1
            return cache_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load cache file {cache_file}: {str(e)}")
    
    def cache_exists(self, image_path: str, timestep: int) -> bool:
        """
        Check if a cache entry exists for the given image and timestep.
        
        Args:
            image_path: Path to the training image
            timestep: The timestep
            
        Returns:
            True if cache exists
        """
        cache_key = self._generate_cache_key(image_path, timestep)
        cache_file = self._get_cache_filepath(cache_key)
        return cache_file.exists()
    
    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache_hits and cache_misses
        """
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
        }
    
    def clear_cache(self) -> int:
        """
        Delete all cache files in the cache directory.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pt"):
            cache_file.unlink()
            count += 1
        
        # Also remove metadata file if it exists
        if self.metadata_file.exists():
            self.metadata_file.unlink()
            count += 1
        
        return count
    
    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Save cache metadata to disk.
        
        Args:
            metadata: Metadata dictionary to save
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self) -> dict[str, Any] | None:
        """
        Load cache metadata from disk.
        
        Returns:
            Metadata dictionary or None if file doesn't exist
        """
        if not self.metadata_file.exists():
            return None
        
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def apply_cfg_regularisation(
        self,
        skip_percentile: float = 75.0,
        dampening_strength: float = 0.5,
    ) -> dict[str, Any]:
        """
        Post-process cached predictions for CFG_REGULARISE target mode.
        
        Uses a single consistent metric throughout: the guidance delta norm
        Δ(i) = ||p_cond(i) - p_empty(i)||_2, which measures how much guidance
        the parent model applies to each sample.
        
        Mathematical consistency:
          - MEASURE guidance:   Δ(i) = ||p_cond - p_empty||_2
          - NORMALIZE guidance: α = mean(Δ) / Δ(i)
          - SCALE guidance:     p_reg = p_empty + α * (p_cfg - p_empty)
        
        All three steps operate on the guidance component. The unconditional
        base prediction p_empty is preserved exactly. This is equivalent to
        training at effective CFG scale α·s, which forces the student to need
        higher CFG at inference without degrading unconditional quality.
        
        Args:
            skip_percentile: Percentile above which samples are marked as skip (0-100)
            dampening_strength: How aggressively to dampen low-guidance samples (0-1)
            
        Returns:
            Statistics dictionary with details about the regularisation
        """
        print(f"\n{'='*80}")
        print(f"CFG REGULARISATION POST-PROCESSING")
        print(f"{'='*80}")
        print(f"Skip percentile: {skip_percentile}")
        print(f"Dampening strength: {dampening_strength}")
        
        # Phase 1: Collect guidance delta norms from all cache files
        cache_files = list(self.cache_dir.glob("*.pt"))
        cache_files = [f for f in cache_files if f.name != "cache_metadata.json"]
        
        if len(cache_files) == 0:
            print("[CFG Regularisation] No cache files found!")
            return {'total': 0, 'skipped': 0, 'dampened': 0}
        
        print(f"[CFG Regularisation] Found {len(cache_files)} cache files")
        
        entries = []
        # Track first-seen per image: only the first measurement per image_path
        # is used for the skip/dampen DECISION. All entries for that image inherit
        # the same decision, but the dampening FACTOR is per-entry (per timestep).
        first_seen_per_image: dict[str, dict] = {}
        
        for cache_file in cache_files:
            try:
                data = torch.load(cache_file, map_location='cpu', weights_only=False)
                predicted = data.get('predicted')
                if predicted is None:
                    continue
                
                guidance_delta = data.get('cfg_regularise_initial_loss')
                image_path = data.get('cfg_regularise_image_path', data.get('image_path', ''))
                
                if guidance_delta is None:
                    print(f"[CFG Regularisation] WARNING: {cache_file.name} missing guidance delta, skipping")
                    continue
                
                entry = {
                    'file': cache_file,
                    'guidance_delta': float(guidance_delta),
                    'image_path': image_path,
                }
                entries.append(entry)
                
                # Record only first-seen measurement per image
                if image_path not in first_seen_per_image:
                    first_seen_per_image[image_path] = entry
                    
            except Exception as e:
                print(f"[CFG Regularisation] Error loading {cache_file}: {e}")
                continue
        
        if len(entries) == 0:
            print("[CFG Regularisation] No valid entries with regularisation metadata!")
            return {'total': 0, 'skipped': 0, 'dampened': 0}
        
        # Phase 2: Compute statistics
        # Skip threshold: based on first-seen measurements (one per image)
        first_seen_deltas = np.array([e['guidance_delta'] for e in first_seen_per_image.values()])
        # Normalisation target: mean guidance delta across ALL entries (per-timestep)
        all_deltas = np.array([e['guidance_delta'] for e in entries])
        
        loss_threshold = float(np.percentile(first_seen_deltas, skip_percentile))
        mean_guidance = float(np.mean(all_deltas))
        
        # Guard against degenerate cases
        if loss_threshold <= 0:
            print("[CFG Regularisation] WARNING: Loss threshold is 0, no samples will be skipped")
            loss_threshold = float(np.max(first_seen_deltas)) + 1.0  # skip nothing
        
        print(f"[CFG Regularisation] Unique images: {len(first_seen_per_image)}")
        print(f"[CFG Regularisation] Total cache entries: {len(entries)}")
        print(f"[CFG Regularisation] Guidance delta stats (first-seen): mean={first_seen_deltas.mean():.6f}, "
              f"std={first_seen_deltas.std():.6f}, min={first_seen_deltas.min():.6f}, max={first_seen_deltas.max():.6f}")
        print(f"[CFG Regularisation] Guidance delta stats (all entries): mean={mean_guidance:.6f}, "
              f"std={all_deltas.std():.6f}, min={all_deltas.min():.6f}, max={all_deltas.max():.6f}")
        print(f"[CFG Regularisation] Skip threshold (p{skip_percentile}): {loss_threshold:.6f}")
        print(f"[CFG Regularisation] Normalisation target (mean Δ): {mean_guidance:.6f}")
        
        # Build per-image skip/dampen DECISION based on first-seen measurement
        image_decisions: dict[str, bool] = {}
        for image_path, first_entry in first_seen_per_image.items():
            image_decisions[image_path] = first_entry['guidance_delta'] > loss_threshold
        
        # Phase 3: Apply regularisation to each cache entry
        skipped_count = 0
        dampened_count = 0
        unchanged_count = 0
        factors = []
        
        for entry in entries:
            cache_file = entry['file']
            data = torch.load(cache_file, map_location='cpu', weights_only=False)
            image_path = entry['image_path']
            should_skip = image_decisions.get(image_path, False)
            
            if should_skip:
                # High guidance need → mark for skipping during training
                data['cfg_regularise_skip'] = True
                data['cfg_regularise_dampening_factor'] = 1.0
                skipped_count += 1
            elif entry['guidance_delta'] > 1e-8:
                # Dampen the guidance component to normalise across samples.
                #
                # Cached prediction: p_cfg = p_empty + s · (p_cond − p_empty)
                # Regularised:       p_reg = p_empty + α · (p_cfg − p_empty)
                #
                # where α brings this sample's guidance strength toward the mean:
                #   ideal_α = mean(Δ) / Δ(i)     [full normalisation]
                #   w = (1 − Δ(i)/threshold) · λ  [dampening weight: 0 at threshold, λ at Δ=0]
                #   α = 1 + w · (ideal_α − 1)     [lerp: w=0→no change, w=1→full normalisation]
                #   α ∈ [0.25, 4.0]                [clamp for stability]
                
                ideal_alpha = mean_guidance / entry['guidance_delta']
                loss_ratio = entry['guidance_delta'] / loss_threshold
                dampening_weight = (1.0 - loss_ratio) * dampening_strength
                
                factor = 1.0 + dampening_weight * (ideal_alpha - 1.0)
                factor = max(0.25, min(4.0, factor))
                
                predicted_empty = data.get('predicted_empty')
                if predicted_empty is not None:
                    guidance = data['predicted'] - predicted_empty
                    data['predicted'] = predicted_empty + factor * guidance
                else:
                    # Fallback: shouldn't happen with CFG cache, but handle gracefully
                    data['predicted'] = data['predicted'] * factor
                
                data['cfg_regularise_skip'] = False
                data['cfg_regularise_dampening_factor'] = factor
                factors.append(factor)
                dampened_count += 1
            else:
                # Near-zero guidance delta — leave unchanged
                data['cfg_regularise_skip'] = False
                data['cfg_regularise_dampening_factor'] = 1.0
                unchanged_count += 1
            
            torch.save(data, cache_file)
        
        factors_arr = np.array(factors) if factors else np.array([1.0])
        stats = {
            'total': len(entries),
            'skipped': skipped_count,
            'dampened': dampened_count,
            'unchanged': unchanged_count,
            'loss_threshold': loss_threshold,
            'mean_guidance_delta': mean_guidance,
            'skip_percentile': skip_percentile,
            'dampening_strength': dampening_strength,
            'factor_mean': float(factors_arr.mean()),
            'factor_std': float(factors_arr.std()),
            'factor_min': float(factors_arr.min()),
            'factor_max': float(factors_arr.max()),
        }
        
        # Save regularisation stats to metadata
        self.save_metadata({
            'cfg_regularisation': stats,
            'parent_model_path': self.parent_model_path,
            'parent_model_type': self.parent_model_type,
            'target_mode': self.target_mode,
            'cfg_scale': self.cfg_scale,
        })
        
        print(f"\n[CFG Regularisation] Results:")
        print(f"  Total samples: {len(entries)}")
        print(f"  Skipped (high guidance need): {skipped_count}")
        print(f"  Dampened (over-guided): {dampened_count}")
        print(f"  Unchanged (near-zero delta): {unchanged_count}")
        print(f"  Dampening factors: mean={factors_arr.mean():.4f}, "
              f"std={factors_arr.std():.4f}, range=[{factors_arr.min():.4f}, {factors_arr.max():.4f}]")
        print(f"{'='*80}\n")
        
        return stats
