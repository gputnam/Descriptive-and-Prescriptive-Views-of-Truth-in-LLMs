"""Model loading, residual-stream capture, and steering.

Every read and every write here is on the OUTPUT of decoder block L
(`model.model.layers[L]`), before the final RMSNorm. Both directions are fit on
this object and steering adds to the same object.
"""
from contextlib import contextmanager

import torch


def load_model(name, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required (the 14B model is ~28 GB in bf16)")
    print(f"[model] loading {name} ({dtype}) on {torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map={"": 0})
    model.eval()
    return model, tok


def residual_layer(model, layer):
    return model.model.layers[layer]


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def last_token_states(model, batches, layer):
    """batches: iterable of LongTensor [B, T] with NO padding (rows of equal length).
    Yields float32 CPU [B, H] states of the last position at `layer`."""
    captured = {}

    def hook(module, inputs, output):
        captured["h"] = _hidden(output)[:, -1, :].detach().to(torch.float32).cpu()

    handle = residual_layer(model, layer).register_forward_hook(hook)
    device = next(model.parameters()).device
    try:
        for ids in batches:
            with torch.inference_mode():
                model.model(input_ids=ids.to(device), use_cache=False)
            yield captured.pop("h")
    finally:
        handle.remove()


def mean_hidden_norm(model, input_ids, layer, start=0, end=None):
    """(sum of ||h||, count) at `layer` over positions [start, end) of one sequence.
    Float32 before norming: bf16 would quantize a ~300-magnitude norm too coarsely."""
    captured = {}

    def hook(module, inputs, output):
        captured["h"] = _hidden(output)[0].detach().to(torch.float32)

    handle = residual_layer(model, layer).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model.model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    norms = captured["h"][start:end].norm(dim=-1)
    return float(norms.sum()), int(norms.numel())


@contextmanager
def steer(model, layer, direction, alphas):
    """Add alphas[i] * direction to row i of block `layer`'s output, at EVERY position
    and on every forward call - the prefill of the re-read prefix and each decode step
    alike. alpha == 0 rows are untouched exactly."""
    p = next(model.parameters())
    v = direction.to(device=p.device, dtype=p.dtype).reshape(-1)
    delta = torch.as_tensor(alphas, device=p.device, dtype=p.dtype).reshape(-1, 1, 1) * v

    def hook(module, inputs, output):
        h = _hidden(output)
        if delta.shape[0] not in (1, h.shape[0]):
            raise RuntimeError(f"steer: {delta.shape[0]} alphas for a batch of {h.shape[0]}")
        h = h + delta
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = residual_layer(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
