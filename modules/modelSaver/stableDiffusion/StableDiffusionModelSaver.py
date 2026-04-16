import copy
import os.path
from pathlib import Path

from modules.model.StableDiffusionModel import StableDiffusionModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_sd_diffusers_to_ckpt import convert_sd_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch

import yaml
from safetensors.torch import save_file


class StableDiffusionModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: StableDiffusionModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")

        if dtype is not None:
            save_pipeline = copy.deepcopy(pipeline)
            save_pipeline.to(device="cpu", dtype=dtype, silence_dtype_warnings=True)
        else:
            save_pipeline = pipeline

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = convert_sd_diffusers_to_ckpt(
            model_type,
            self._get_dequantized_state_dict(model.vae),
            self._get_dequantized_state_dict(model.unet),
            self._get_dequantized_state_dict(model.text_encoder),
            model.noise_scheduler
        )
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

        yaml_name = os.path.splitext(destination)[0] + '.yaml'
        with open(yaml_name, 'w', encoding='utf8') as f:
            yaml.dump(model.sd_config, f, default_flow_style=False, allow_unicode=True)

    def __save_internal(
            self,
            model: StableDiffusionModel,
            destination: str,
    ):
            from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
            from safetensors.torch import save_file as safetensors_save_file
            import json as _json

            self.__save_diffusers(model, destination, None)

            components = [
                ("unet", model.unet),
                ("vae", model.vae),
                ("text_encoder", model.text_encoder),
            ]

            for component_name, module in components:
                if not any(isinstance(m, QuantizedLinearMixin) for m in module.modules()):
                    continue

                component_dir = os.path.join(destination, component_name)
                if not os.path.isdir(component_dir):
                    continue

                state_dict = self._get_dequantized_state_dict(module)
                state_dict = {k: v.contiguous() for k, v in state_dict.items()}

                index_file = os.path.join(component_dir, "diffusion_pytorch_model.safetensors.index.json")
                if os.path.isfile(index_file):
                    with open(index_file) as f:
                        index = _json.load(f)
                    shard_files: dict[str, list[str]] = {}
                    for key, filename in index["weight_map"].items():
                        shard_files.setdefault(filename, []).append(key)
                    for filename, keys in shard_files.items():
                        shard = {k: state_dict[k] for k in keys if k in state_dict}
                        if shard:
                            safetensors_save_file(shard, os.path.join(component_dir, filename))
                else:
                    for filename in ["diffusion_pytorch_model.safetensors", "model.safetensors"]:
                        filepath = os.path.join(component_dir, filename)
                        if os.path.isfile(filepath):
                            safetensors_save_file(state_dict, filepath)
                            break

    def save(
            self,
            model: StableDiffusionModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, model_type, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
