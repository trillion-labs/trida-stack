"""The guard that keeps fetched PolyForm-NC kernels out of git — no network."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_supply_chain",
                                              Path(__file__).parent / "check_supply_chain.py")
csc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csc)


def test_a_tracked_noncommercial_kernel_is_flagged_wherever_it_lives():
    tracked = [
        "inference/zoro-1/vllm/vllm_native_diffusion/block_causal_readout.py",
        "inference/vllm/vllm_native_diffusion/block_causal_readout.py",     # left behind by the move
        "train/block_gated_delta_rule/chunk.py",
    ]
    assert csc.tracked_noncommercial(tracked) == tracked


def test_the_recipes_and_first_party_files_are_allowed():
    tracked = [
        "train/block_gated_delta_rule/fetch_kernels.sh",
        "train/block_gated_delta_rule/trillion_mods.patch",
        "train/block_gated_delta_rule/README.md",
        "inference/zoro-1/vllm/vllm_native_diffusion/KERNELS.md",
        "inference/zoro-1/vllm/vllm_native_diffusion/plugin.py",
        "train/hf_block_diffusion_hybrid.py",
    ]
    assert csc.tracked_noncommercial(tracked) == []


def test_the_repository_tracks_none():
    assert csc.tracked_noncommercial(csc.git_tracked()) == []


def test_gitignore_hides_the_vllm_kernel_at_any_path():
    import subprocess
    for path in ("inference/zoro-1/vllm/vllm_native_diffusion/block_causal_readout.py",
                 "inference/vllm/vllm_native_diffusion/block_causal_readout.py"):
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=csc.ROOT)
        assert r.returncode == 0, f"{path} is not git-ignored"
