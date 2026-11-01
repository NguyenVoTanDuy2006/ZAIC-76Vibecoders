from __future__ import annotations


def count_model_parameters(adapter) -> int:
    """Count unique learned tensors in the ASR model, excluding wrapper and processor."""
    model = adapter.model
    if not hasattr(model, "parameters"):
        model = model.model
    return sum(parameter.numel() for parameter in model.parameters())
