
"""LoRA adapters for GPT-2 attention projections."""

import math

import torch
from torch import nn
from safetensors.torch import save_file, load_file


class LoRAConv1D(nn.Module):
    """
    LoRA wrapper for GPT-2's Conv1D projection.

    The pretrained projection remains frozen.
    Only the low-rank matrices are trained.
    """

    def __init__(self, base_layer, rank=4, alpha=8):
        super().__init__()

        if rank < 1:
            raise ValueError("rank must be positive")

        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        in_features, out_features = (
            base_layer.weight.shape
        )

        self.lora_A = nn.Parameter(
            torch.empty(in_features, rank)
        )

        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features)
        )

        nn.init.kaiming_uniform_(
            self.lora_A, a=math.sqrt(5)
        )

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        base_output = self.base_layer(x)

        lora_output = (
            x @ self.lora_A @ self.lora_B
        ) * self.scaling

        return base_output + lora_output


def inject_lora(model, rank=4, alpha=8):
    """Inject LoRA into GPT-2 attention projections."""
    from transformers.pytorch_utils import Conv1D

    # Freeze all pretrained parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced = 0

    for block in model.transformer.h:
        attention = block.attn

        for name in ("c_attn", "c_proj"):
            layer = getattr(attention, name)

            if not isinstance(layer, Conv1D):
                raise TypeError(
                    f"Expected GPT-2 Conv1D for {name}"
                )

            setattr(
                attention,
                name,
                LoRAConv1D(layer, rank, alpha)
            )

            replaced += 1

    return replaced


def count_parameters(model):
    total = sum(
        p.numel() for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "percentage": 100 * trainable / total
    }


def save_lora_adapter(model, path):
    """Save only the trainable LoRA parameters."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    adapter_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }

    if not adapter_state:
        raise ValueError("No LoRA parameters found")

    save_file(adapter_state, str(path))


def load_lora_adapter(model, path):
    """Load saved LoRA parameters into an injected model."""
    adapter_state = load_file(str(path))
    parameters = dict(model.named_parameters())

    for name, tensor in adapter_state.items():
        if name not in parameters:
            raise KeyError(f"Missing parameter: {name}")

        parameter = parameters[name]

        if parameter.shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {name}"
            )

        with torch.no_grad():
            parameter.copy_(
                tensor.to(
                    device=parameter.device,
                    dtype=parameter.dtype
                )
            )

    return model

def merge_lora_weights(model):
    """
    Merge LoRA matrices into GPT-2 Conv1D weights.

    After merging, LoRA wrappers are removed.
    """
    import torch

    merged_layers = 0

    for block in model.transformer.h:
        attention = block.attn

        for name in ("c_attn", "c_proj"):
            layer = getattr(attention, name)

            if not isinstance(layer, LoRAConv1D):
                continue

            base_layer = layer.base_layer

            # GPT-2 Conv1D stores weights as
            # (input_features, output_features).
            delta_weight = (
                layer.lora_A @ layer.lora_B
            ) * layer.scaling

            with torch.no_grad():
                base_layer.weight.add_(
                    delta_weight.to(
                        device=base_layer.weight.device,
                        dtype=base_layer.weight.dtype
                    )
                )

            # Replace the LoRA wrapper with
            # the updated original projection.
            setattr(
                attention,
                name,
                base_layer
            )

            merged_layers += 1

    if merged_layers == 0:
        raise ValueError(
            "No LoRA layers were found to merge"
        )

    return merged_layers
