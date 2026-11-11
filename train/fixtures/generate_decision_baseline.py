"""Generate frozen regression values from the pre-refactor 138239d code only."""

import ast, json, subprocess, sys, types
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path.cwd()))
REV = "138239df5c5a305f74cc1dee7b1de206d8ef6438"


def read(path):
    return subprocess.check_output(["git", "show", f"{REV}:{path}"], text=True)


legacy = types.ModuleType("train.jev")
exec(compile(read("train/jev.py"), "138239d:train/jev.py", "exec"), legacy.__dict__)
sys.modules["train.jev"] = legacy
import train

train.jev = legacy
# Extract original methods/helpers without loading transformers or model weights.
ns = {"torch": torch}
tree = ast.parse(read("train/hf_block_diffusion_hybrid.py"))
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in [
        "_cu_seqlens_from_seg",
        "_seq_idx_from_seg",
        "_positions_from_seg",
    ]:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "138239d:hybrid", "exec"), ns)
cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HFBlockDiffusionHybrid"
)
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_jev")
exec(compile(ast.Module(body=[method], type_ignores=[]), "138239d:forward_jev", "exec"), ns)
base = ast.parse(read("train/hf_block_diffusion.py"))
basecls = next(n for n in base.body if isinstance(n, ast.ClassDef) and n.name == "HFBlockDiffusion")
groups = next(
    n for n in basecls.body if isinstance(n, ast.FunctionDef) and n.name == "param_groups"
)
exec(compile(ast.Module(body=[groups], type_ignores=[]), "138239d:param_groups", "exec"), ns)


class Model(torch.nn.Module):
    forward_jev = ns["forward_jev"]
    param_groups = ns["param_groups"]

    def __init__(self):
        super().__init__()
        self.jev_head = legacy.PointerHead(4, 2)
        self.tm = SimpleNamespace(
            embed_tokens=torch.nn.Embedding(10, 4),
            norm=torch.nn.Identity(),
            rotary_emb=lambda *a, **kw: None,
        )
        self.compute_dtype, self.is_vl, self.grad_checkpoint = torch.float32, False, False
        self._ar_doc_mask = lambda *a, **kw: (None, "eager")
        self._set_attn_impl = lambda *a: None
        self._run_decoder_stack_ar = lambda hidden, *a, **kw: hidden


def pack(value):
    if isinstance(value, torch.Tensor):
        return {"tensor": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    return value


torch.manual_seed(0)
model = Model()
batch = {
    "input_ids": torch.tensor([[1, 2, 3]]),
    "seg_id": torch.tensor([[0, 0, 0]]),
    "decide": torch.tensor([2]),
    "opts": torch.tensor([0, 1]),
    "owner": torch.tensor([0, 0]),
    "questions": [{"start": 0, "k": 2, "variant": 0, "label": 1}],
}
initial = {k: v.clone() for k, v in model.state_dict().items()}
loss, logs = model.forward_jev(batch)
loss.backward()
param_names = {id(p): n for n, p in model.named_parameters()}
groups = model.param_groups(jev_head_lr_scale=5)
report = {
    "revision": subprocess.check_output(["git", "rev-parse", REV], text=True).strip(),
    "seed": 0,
    "torch_version": torch.__version__,
    "batch": pack(batch),
    "head": pack(initial),
    "embedding": pack(model.tm.embed_tokens.weight),
    "loss": loss.item(),
    "gradients": pack({n: p.grad for n, p in model.named_parameters()}),
    "groups": [
        {"lr_scale": g["lr_scale"], "parameters": [param_names[id(p)] for p in g["params"]]}
        for g in groups
    ],
}
optim = torch.optim.AdamW(groups, lr=0.001)
optim.step()
report["updated_head"] = pack(model.state_dict())
report["optimizer"] = pack(optim.state_dict())
out = Path("train/fixtures/decision_138239d.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print("Baseline loss", report["loss"], "groups", report["groups"])
