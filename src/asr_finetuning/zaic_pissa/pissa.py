"""PiSSA lifecycle adapted from Research; exact residual resume, portable export."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch


class PiSSAManager:
    def __init__(self, settings):
        self.settings = copy.deepcopy(settings)
        self.model = None
        self.initial = {}
        self.residual = {}
        self.initialization_path = None

    def attach(self, base, initialization_state=None):
        from peft import LoraConfig, get_peft_model

        if self.model is not None or getattr(base, "peft_config", None):
            raise ValueError("PiSSA requires a fresh original base")
        s = self.settings
        names = [
            n
            for n, m in base.named_modules()
            if n.split(".")[-1] in s["target_modules"] and type(m) is torch.nn.Linear
        ]
        audio = [n for n in names if "audio" in n or "encoder" in n]
        if not audio or len(audio) == len(names):
            raise ValueError("q/v targets must include audio encoder and decoder")
        for name in names:
            if s["r"] > min(base.get_submodule(name).weight.shape):
                raise ValueError("rank exceeds target dimensions")
        if initialization_state and initialization_state["settings"] != s:
            raise ValueError("PiSSA resume settings differ")
        devices = sorted(
            {p.device.index for p in base.parameters() if p.device.type == "cuda"}
        )
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(s["init_seed"])
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(s["init_seed"])
            self.model = get_peft_model(
                base,
                LoraConfig(
                    r=s["r"],
                    lora_alpha=s["alpha"],
                    lora_dropout=s["dropout"],
                    target_modules=names,
                    bias="none",
                    task_type=None,
                    init_lora_weights=True
                    if initialization_state
                    else s["init_lora_weights"],
                ),
            )
        state = self.model.state_dict()
        keys = {k for k in state if ".lora_A.default." in k or ".lora_B.default." in k}
        residual = {
            k.replace(".lora_A.default.weight", ".base_layer.weight")
            for k in keys
            if ".lora_A." in k
        }
        if initialization_state:
            if (
                set(initialization_state["initial"]) != keys
                or set(initialization_state["residual"]) != residual
            ):
                raise ValueError("PiSSA initialization/residual keys differ")
            with torch.no_grad():
                for k, v in {
                    **initialization_state["initial"],
                    **initialization_state["residual"],
                }.items():
                    if state[k].shape != v.shape or state[k].dtype != v.dtype:
                        raise ValueError(f"PiSSA shape/dtype mismatch: {k}")
                    state[k].copy_(v.to(state[k].device))
            self.model.peft_config["default"].init_lora_weights = s["init_lora_weights"]
        self.initial = {k: state[k].detach().cpu().clone() for k in keys}
        self.residual = {k: state[k].detach().cpu().clone() for k in residual}
        return self.model

    def state_dict(self):
        current = self.model.state_dict()
        return {
            "settings": self.settings,
            "initial": self.initial,
            "residual": self.residual,
            "adapter": {k: current[k].detach().cpu().clone() for k in self.initial},
        }

    def restore_adapter(self, state):
        current = self.model.state_dict()
        if set(state["adapter"]) != set(self.initial):
            raise ValueError("native adapter keys differ")
        with torch.no_grad():
            for k, v in state["adapter"].items():
                if current[k].shape != v.shape or current[k].dtype != v.dtype:
                    raise ValueError("native adapter shape/dtype differs")
                current[k].copy_(v.to(current[k].device))

    def save_initialization(self, destination):
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file

        destination = Path(destination)
        if destination.name == "default":
            raise ValueError("initial adapter directory must not be named default")
        if destination.exists():
            actual = load_file(str(destination / "adapter_model.safetensors"))
            expected = get_peft_model_state_dict(self.model, state_dict=self.initial)
            if actual.keys() != expected.keys() or any(
                not torch.equal(actual[k], expected[k]) for k in actual
            ):
                raise ValueError("existing PiSSA initialization differs")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=".pissa-initial-", dir=destination.parent)
            )
            config = self.model.peft_config["default"]
            old = config.init_lora_weights
            try:
                config.init_lora_weights = True
                self.model.save_pretrained(
                    staging,
                    state_dict=self.initial,
                    safe_serialization=True,
                    save_embedding_layers=False,
                )
                os.replace(staging, destination)
            finally:
                config.init_lora_weights = old
                if staging.exists():
                    shutil.rmtree(staging)
        self.initialization_path = destination.resolve()

    def export(self, destination, processor=None):
        if self.initialization_path is None:
            raise RuntimeError("save initialization before export")
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".adapter-", dir=destination.parent))
        model = self.model
        configs = copy.deepcopy(model.peft_config)
        active = model.active_adapter
        flags = {n: m.training for n, m in model.named_modules()}
        gradients = {n: p.requires_grad for n, p in model.named_parameters()}
        devices = sorted(
            {p.device.index for p in model.parameters() if p.device.type == "cuda"}
        )
        try:
            with torch.random.fork_rng(devices=devices):
                model.save_pretrained(
                    staging,
                    selected_adapters=["default"],
                    safe_serialization=True,
                    save_embedding_layers=False,
                    path_initial_model_for_weight_conversion=str(
                        self.initialization_path
                    ),
                )
            if processor is not None:
                processor.save_pretrained(staging)
            (staging / "training_method.json").write_text(
                json.dumps(
                    {
                        "method": "pissa",
                        "export": "converted_standard_lora",
                        "training_rank": self.settings["r"],
                        "export_rank": 2 * self.settings["r"],
                        "load_on": "original_pretrained_base",
                        "settings": self.settings,
                    },
                    indent=2,
                )
                + "\n"
            )
            os.replace(staging, destination)
        finally:
            for name, config in configs.items():
                model.peft_config[name] = config
            model.set_adapter(active)
            for n, p in model.named_parameters():
                if n in gradients:
                    p.requires_grad_(gradients[n])
            for n, m in model.named_modules():
                if n in flags:
                    m.training = flags[n]
            if staging.exists():
                shutil.rmtree(staging)
        return destination
