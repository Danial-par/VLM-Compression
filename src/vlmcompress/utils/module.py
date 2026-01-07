from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


def get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """Return (parent_module, attribute_name) for a dotted module path."""
    if module_name == "" or module_name is None:
        raise ValueError("module_name must be a non-empty dotted path")
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, attr = get_parent_module(root, module_name)
    setattr(parent, attr, new_module)


def get_module(root: nn.Module, module_name: str) -> nn.Module:
    parent, attr = get_parent_module(root, module_name)
    return getattr(parent, attr)


def any_regex_match(text: str, patterns: Optional[Iterable[str]]) -> bool:
    if patterns is None:
        return False
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def iter_named_linears(
    model: nn.Module,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Iterator[Tuple[str, nn.Linear]]:
    """Yield (module_name, nn.Linear) filtered by regex patterns over module_name."""
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if include and not any_regex_match(name, include):
            continue
        if exclude and any_regex_match(name, exclude):
            continue
        yield name, module
