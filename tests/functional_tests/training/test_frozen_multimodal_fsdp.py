# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real two-rank regressions for frozen parameters and uneven multimodal FSDP execution."""

import argparse
import copy
import datetime
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

_RESULT_PREFIX = "FROZEN_MULTIMODAL_FSDP_RESULT "


class _TinyVisionTower(nn.Module):
    """Frozen conditional tower with layer-granularity FSDP structure."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the frozen tower.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        for layer in self.layers:
            hidden_states = torch.tanh(layer(hidden_states))
        return hidden_states


class _TinyConditionalVLM(nn.Module):
    """Tiny model whose vision branch may execute on only part of the DP group."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.vision_tower = _TinyVisionTower()
        self.output = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor, *, use_media: bool) -> torch.Tensor:
        """Run the conditional multimodal model.

        Args:
            inputs: Tensor of shape [batch, sequence, hidden].
            use_media: Whether to execute the frozen vision branch.

        Returns:
            Tensor of shape [batch, sequence, 1].
        """
        hidden_states = inputs
        for layer in self.layers:
            hidden_states = torch.relu(layer(hidden_states))
        if use_media:
            hidden_states = hidden_states + self.vision_tower(hidden_states)
        return self.output(hidden_states)


def _full_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if isinstance(value, DTensor):
            value = value.full_tensor()
        state_dict[name] = value.detach().clone()
    return state_dict


def _run_training_case(
    *,
    policy: str,
    media_by_rank: tuple[bool, bool],
    device: torch.device,
) -> tuple[list[float], dict[str, torch.Tensor]]:
    torch.manual_seed(1234)
    model = _TinyConditionalVLM().to(device)
    model.vision_tower.requires_grad_(False)

    world_size = dist.get_world_size()
    mesh = init_device_mesh(
        "cuda",
        (1, world_size, 1, 1),
        mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"),
    )
    fp32_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    model = DefaultParallelizationStrategy().parallelize(
        model,
        mesh,
        mp_policy=fp32_policy,
        frozen_multimodal_sharding=policy,
    )

    vision_layer_is_fsdp = isinstance(model.vision_tower.layers[0], FSDPModule)
    assert vision_layer_is_fsdp is (policy == "per_layer")

    optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for microbatch_idx in range(2):
        model.set_requires_gradient_sync(microbatch_idx == 1)
        generator = torch.Generator(device=device).manual_seed(9000 + dist.get_rank() * 10 + microbatch_idx)
        inputs = torch.randn(2, 4, 8, device=device, generator=generator)
        loss = model(inputs, use_media=media_by_rank[dist.get_rank()]).float().square().mean()
        losses.append(loss.detach().item())
        (loss / 2).backward()

    optimizer.step()
    state_dict = _full_state_dict(model)
    for tensor in state_dict.values():
        assert torch.isfinite(tensor).all()
    dist.barrier()
    return losses, state_dict


def _assert_state_dicts_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    """Assert that two full model state dictionaries match.

    Args:
        actual: Mapping of parameter names to tensors of arbitrary parameter
            shapes from the tested policy.
        expected: Mapping with the same keys and parameter shapes from the
            reference policy.
    """
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6, msg=lambda msg: f"{name}: {msg}")


def _run_worker() -> None:
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError(f"This regression requires exactly 2 ranks, got {dist.get_world_size()}.")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # Rank 0 executes the frozen tower while rank 1 skips it. Root ownership
        # must keep FSDP collective ordering aligned through two accumulated
        # microbatches.
        _run_training_case(policy="root", media_by_rank=(True, False), device=device)

        # All ranks skip separately sharded tower layers. The narrow accumulated
        # grad guard must tolerate their never-created lazy unsharded state.
        _run_training_case(policy="per_layer", media_by_rank=(False, False), device=device)

        # Replication has no tower collectives, so it also supports rank-asymmetric
        # media execution while leaving the outer root's collectives aligned.
        _run_training_case(policy="replicate", media_by_rank=(True, False), device=device)

        # Under the documented uniform-execution contract, changing only the
        # frozen tower topology must preserve the optimizer result.
        _, root_state = _run_training_case(policy="root", media_by_rank=(True, True), device=device)
        _, per_layer_state = _run_training_case(policy="per_layer", media_by_rank=(True, True), device=device)
        _assert_state_dicts_close(per_layer_state, root_state)

        if dist.get_rank() == 0:
            print(_RESULT_PREFIX + "PASS", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices")
def test_frozen_multimodal_fsdp_two_rank_regressions() -> None:
    """Root/replicate handle asymmetry; per-layer handles uniform skip and preserves updates."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    repo_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout
    assert _RESULT_PREFIX + "PASS" in completed.stdout


def _dsv41_uneven_vision_worker(
    rank: int,
    rendezvous: str,
    image_counts: tuple[int, int],
    checkpointing: bool,
) -> None:
    """Compare uneven-image FSDP outputs, gradients and an SGD step with the reference."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import (
        DeepseekV41Config,
        DeepseekV41TextConfig,
        DeepseekV41VisionConfig,
    )
    from nemo_automodel.components.models.deepseek_v41.fsdp import fully_shard_deepseek_v41
    from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.manual_seed(17)
    config = DeepseekV41Config(
        text_config=DeepseekV41TextConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            engram_layer_ids=[],
            dtype="float32",
            moe_intermediate_size=8,
            num_attention_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            o_groups=1,
            hc_mult=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            candidate_source_layer_id=-1,
            num_nextn_predict_layers=0,
            dspark_block_size=0,
            dspark_noise_token_id=0,
        ),
        vision_config=DeepseekV41VisionConfig(
            num_hidden_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            intermediate_size=12,
            patch_size=2,
            downsample_ratio=2,
        ),
        image_token_id=1,
        dtype="float32",
    )
    reference = DeepseekV41ForCausalLM(
        config,
        backend=BackendConfig(
            attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch"
        ),
    ).to(device)
    reference.initialize_weights(device, dtype=torch.float32)
    # Delimiters are frozen for LoRA, just as in the RL recipe.
    for name in ("image_start", "image_end", "image_newline"):
        getattr(reference.model, name).requires_grad_(False)
    distributed = copy.deepcopy(reference)
    if checkpointing:
        for i, block in enumerate(distributed.model.vision.blocks):
            distributed.model.vision.blocks[i] = checkpoint_wrapper(block)
    count = image_counts[rank]
    ids = torch.tensor([[2] + [1] * (4 * count) + [3]], device=device)
    types = torch.tensor([[-1] + [0, 1, 2, 3] * count + [-1]], device=device)
    pixels = torch.linspace(-1, 1, max(1, count * 4 * 3 * 2 * 2), device=device)
    pixels = pixels[: count * 48].reshape(count * 4, 3, 2, 2)
    grids = torch.tensor([[2, 2]] * count, device=device, dtype=torch.long).reshape(count, 2)
    model_inputs = {"input_ids": ids}
    if count:
        model_inputs.update(pixel_values=pixels, image_grid_hws=grids, vision_token_types=types)
    expected = reference(**model_inputs).logits
    expected.square().sum().backward()

    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cuda", (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        fully_shard_deepseek_v41(distributed.model.vision, mesh=mesh, mp_policy=policy)
        fully_shard(distributed.model.aligner, mesh=mesh, mp_policy=policy)
        fully_shard(distributed.model, mesh=mesh, mp_policy=policy)
        fully_shard(distributed, mesh=mesh, mp_policy=policy)
        optimizer = torch.optim.SGD(distributed.parameters(), lr=0.01)
        with torch.no_grad():
            torch.testing.assert_close(distributed(**model_inputs).logits, expected.detach(), rtol=1e-5, atol=1e-5)
        actual = distributed(**model_inputs).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        actual.square().sum().backward()
        ref_params = dict(reference.named_parameters())
        expected_parameters = {}
        for name, parameter in distributed.named_parameters():
            if not name.startswith(("model.vision.", "model.aligner.")):
                continue
            ref_name = name.replace("._checkpoint_wrapped_module", "")
            wanted = ref_params[ref_name].grad
            wanted = torch.zeros_like(ref_params[ref_name]) if wanted is None else wanted.clone()
            dist.all_reduce(wanted)
            wanted /= 2
            assert parameter.grad is not None, name
            actual_grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
            torch.testing.assert_close(actual_grad, wanted, rtol=1e-4, atol=1e-5, msg=name)
            expected_parameters[name] = ref_params[ref_name].detach() - 0.01 * wanted
        optimizer.step()
        for name, parameter in distributed.named_parameters():
            if name in expected_parameters:
                actual_parameter = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
                torch.testing.assert_close(actual_parameter, expected_parameters[name], rtol=1e-4, atol=1e-5, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.timeout(120)
@pytest.mark.parametrize("image_counts,checkpointing", [((2, 0), False), ((2, 1), True)])
def test_uneven_vision_fsdp_matches_unsharded_forward_and_gradients(
    tmp_path: Path, image_counts: tuple[int, int], checkpointing: bool
) -> None:
    """A rank with no images must participate in the same vision collectives."""
    if torch.cuda.device_count() < 2:
        pytest.skip("This FSDP regression test requires two CUDA devices")
    mp.spawn(
        _dsv41_uneven_vision_worker,
        args=(str(tmp_path / "rendezvous"), image_counts, checkpointing),
        nprocs=2,
        join=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.worker:
        _run_worker()
