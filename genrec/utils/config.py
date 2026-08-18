from __future__ import annotations

import yaml
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, List

import re
from typing import Any
from dataclasses import is_dataclass, asdict

from genrec.option.options import Options  # your existing dataclass for model options


class DotDict(dict):
    """
    Dictionary with attribute-style access.
    Nested dicts are converted to DotDict recursively.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict):
                self[k] = DotDict(v)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _load_yaml(path: str | Path) -> DotDict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return DotDict(data or {})


@dataclass
class TrainingArgs:
    epochs: int = 100
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float | None = None
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    log_every: int = 20
    eval_every: int = 1000


def _dataclass_from_section(cls, section: Any) -> Any:
    """
    Generic helper: instantiate a dataclass `cls` from a config section (DotDict/dict).
    Only fields that exist in the dataclass are used.
    """
    if section is None:
        section = {}
    # section is expected to behave like a dict
    kwargs = {}
    for f in fields(cls):
        if isinstance(section, dict) and f.name in section:
            kwargs[f.name] = section[f.name]
    return cls(**kwargs)


def build_options_from_config(cfg: DotDict, opts_cls: type = Options) -> Options:
    opt = opts_cls()
    model_section = getattr(cfg, "model", DotDict())

    for f in fields(opt):
        if f.name in model_section:
            value = model_section[f.name]

            if isinstance(getattr(opt, f.name), dict) and isinstance(value, dict):
                getattr(opt, f.name).update(value)
            else:
                setattr(opt, f.name, value)

    # Rebuild computed attributes (in_channels, unet_from_pretrained_kwargs)
    # so they reflect the updated field values
    opt.__post_init__()

    # Apply any explicit unet_from_pretrained_kwargs overrides from config on top
    cfg_unet_kwargs = model_section.get("unet_from_pretrained_kwargs", None)
    if isinstance(cfg_unet_kwargs, dict):
        opt.unet_from_pretrained_kwargs.update(cfg_unet_kwargs)

    return opt



def build_training_args_from_config(cfg: DotDict) -> TrainingArgs:
    """
    Build TrainingArgs from cfg.train.
    """
    train_section = getattr(cfg, "train", DotDict())
    return _dataclass_from_section(TrainingArgs, train_section)


def apply_overrides(cfg: DotDict, overrides: List[str]) -> None:
    """
    Simple dotted-key overrides:

        train.lr=5e-5
        train.batch_size=2
        model.hidden_dim=256

    modifies cfg in-place.
    """
    for ov in overrides or []:
        if "=" not in ov:
            continue
        key, value_str = ov.split("=", 1)

        # Try to infer type (bool / int / float / str)
        v_lower = value_str.lower()
        if v_lower in ("true", "false"):
            value: Any = (v_lower == "true")
        else:
            try:
                if "." in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
            except ValueError:
                value = value_str

        # Descend through nested keys
        parts = key.split(".")
        obj: Any = cfg
        for p in parts[:-1]:
            if not hasattr(obj, p):
                # create intermediate DotDict if missing
                setattr(obj, p, DotDict())
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)


def load_experiment_config(
    path: str | Path,
    overrides: List[str] | None = None,
    opts_cls: type = Options,
) -> tuple[DotDict, Options, TrainingArgs]:
    """
    Load YAML config, apply optional overrides, and return:

        full_cfg:    DotDict of the whole config tree
        model_opts:  Options dataclass for the model
        train_args:  TrainingArgs dataclass for the training loop
    """
    cfg = _load_yaml(path)
    apply_overrides(cfg, overrides or [])

    model_opts = build_options_from_config(cfg, opts_cls=opts_cls)
    train_args = build_training_args_from_config(cfg)

    return cfg, model_opts, train_args



# ANSI colors
CYAN = "\033[96m"
MAGENTA = "\033[95m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
WHITE = "\033[97m"
RESET = "\033[0m"


def _to_primitive(obj: Any) -> Any:
    """
    Convert DotDict / dataclass / nested structures into plain python
    structures (dict, list, primitives).
    """
    if is_dataclass(obj):
        return {k: _to_primitive(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_primitive(v) for v in obj]
    return obj


def _indent(text: str, level: int) -> str:
    return "  " * level + text


def _color_value(v: Any) -> str:
    """
    Colorize different value types.
    """
    if isinstance(v, bool):
        return GREEN + str(v).lower() + RESET
    if isinstance(v, (int, float)):
        return YELLOW + str(v) + RESET
    if isinstance(v, str):
        return MAGENTA + f"\"{v}\"" + RESET
    if v is None:
        return WHITE + "null" + RESET
    # fallback
    return WHITE + str(v) + RESET


def _format_yaml(obj: Any, level: int = 0) -> str:
    """
    Recursively pretty-print Python structures as colored YAML-like text.
    """
    if isinstance(obj, dict):
        lines = []
        for key, value in obj.items():
            colored_key = CYAN + str(key) + RESET
            if isinstance(value, (dict, list)):
                lines.append(_indent(f"{colored_key}:", level))
                lines.append(_format_yaml(value, level + 1))
            else:
                colored_val = _color_value(value)
                lines.append(_indent(f"{colored_key}: {colored_val}", level))
        return "\n".join(lines)

    elif isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(_indent(f"-", level))
                lines.append(_format_yaml(item, level + 1))
            else:
                colored_val = _color_value(item)
                lines.append(_indent(f"- {colored_val}", level))
        return "\n".join(lines)

    # primitive fallback
    return _indent(_color_value(obj), level)


def beautify_config(obj: Any) -> str:
    """
    Public entry point:
      Converts DotDict + dataclasses into colored, YAML-ish pretty text.
    """
    primitive = _to_primitive(obj)
    return _format_yaml(primitive, level=0)

