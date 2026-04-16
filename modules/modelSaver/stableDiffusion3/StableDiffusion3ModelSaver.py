import copy
import os.path
from pathlib import Path

from modules.model.StableDiffusion3Model import StableDiffusion3Model
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert.convert_sd3_diffusers_to_ckpt import convert_sd3_diffusers_to_ckpt
from modules.util.enum.ModelFormat import ModelFormat

import torch

from transformers import T5EncoderModel

from safetensors.torch import save_file


class StableDiffusion3ModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: StableDiffusion3Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        if dtype is not None:
            # replace the tokenizers __deepcopy__ before calling deepcopy, to prevent a copy being made.
            # the tokenizer tries to reload from the file system otherwise
            tokenizer_3 = pipeline.tokenizer_3
            tokenizer_3.__deepcopy__ = lambda memo: tokenizer_3

            save_pipeline = copy.deepcopy(pipeline)
            save_pipeline.to(device="cpu", dtype=dtype, silence_dtype_warnings=True)

            delattr(tokenizer_3, '__deepcopy__')
        else:
            save_pipeline = pipeline

        text_encoder_3 = save_pipeline.text_encoder_3
        if text_encoder_3 is not None:
            text_encoder_3_save_pretrained = text_encoder_3.save_pretrained
            def save_pretrained_t5(
                    self,
                    *args,
                    **kwargs,
            ):
                # Saving a safetensors file copies all tensors in RAM.
                # Setting the max_shard_size to 2GB reduces this memory overhead a bit.
                # This parameter is set by patching the function, because it's not exposed to the pipeline.
                kwargs = dict(kwargs)
                kwargs['max_shard_size'] = '2GB'
                text_encoder_3_save_pretrained(*args, **kwargs)

            text_encoder_3.save_pretrained = save_pretrained_t5.__get__(text_encoder_3, T5EncoderModel)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if text_encoder_3 is not None:
            text_encoder_3.save_pretrained = text_encoder_3_save_pretrained

        if dtype is not None:
            del save_pipeline

    def __save_safetensors(
            self,
            model: StableDiffusion3Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = convert_sd3_diffusers_to_ckpt(
            self._get_dequantized_state_dict(model.vae),
            self._get_dequantized_state_dict(model.transformer),
            self._get_dequantized_state_dict(model.text_encoder_1) if model.text_encoder_1 is not None else None,
            self._get_dequantized_state_dict(model.text_encoder_2) if model.text_encoder_2 is not None else None,
            self._get_dequantized_state_dict(model.text_encoder_3) if model.text_encoder_3 is not None else None,
        )
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: StableDiffusion3Model,
            destination: str,
    ):
            from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
            from safetensors.torch import save_file as safetensors_save_file
            import json as _json

            self.__save_diffusers(model, destination, None)

            components = [
                ("transformer", model.transformer),
                ("vae", model.vae),
            ]
            if model.text_encoder_1 is not None:
                components.append(("text_encoder", model.text_encoder_1))
            if model.text_encoder_2 is not None:
                components.append(("text_encoder_2", model.text_encoder_2))
            if model.text_encoder_3 is not None:
                components.append(("text_encoder_3", model.text_encoder_3))

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
            model: StableDiffusion3Model,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
