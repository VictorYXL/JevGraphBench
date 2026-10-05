"""Pinned, local-only SemIf scoring adapter for the proposal rerun."""

from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from src.clients.base import ClientClosedError
from src.utils import paired_graph_ablation as shared


ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / (
    "output/experiments/synthetic/matrix/full-matrix-trained-20260925-v1/"
    "frozen-source/scripts/semif_vllm_bridge.py"
)
BRIDGE_SHA256 = "5cbebc5ed0be03643c9e0d441f4bcd79d7b868d69e1f0606462314cff2cde3a0"
CPU_REFERENCE = ROOT / "output/experiments/diagnostics/native/semif-cpu-pilot-20260924-v1"
MODEL = Path("/fastdata/xianya/models/Qwen3.5-4B")


def _load_bridge():
    if sha256(BRIDGE.read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError("Preserved SemIf bridge source hash differs")
    spec = importlib.util.spec_from_file_location("proposal_pinned_semif", BRIDGE)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load preserved SemIf bridge")
    module = importlib.util.module_from_spec(spec)
    previous_path = sys.path[:]
    previous_bytecode = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
    return module


def create_client():
    """Keep historical scoring/encoding; migrate only the coordination import."""
    bridge = _load_bridge()
    tokenizer, encoder = bridge.upstream_functions(CPU_REFERENCE, MODEL)

    class ProposalScoringClient(bridge.SemIfVLLMBridge):
        async def initialize(self):
            if self._closed:
                raise ClientClosedError("Bridge is closed")
            if self._ready:
                return
            self._guard = shared.lane_locks(["qwen4"])
            self._guard.__enter__()
            self._acquired = True
            try:
                config = shared.model_configs(models=("qwen4",))["qwen4"]
                if config.base_url != bridge.BASE_URL or config.model != bridge.MODEL_ID:
                    raise ValueError("Shared qwen4 configuration differs")
                await shared.vllm_preflight([SimpleNamespace(model=config)])
                self._ready = True
            except BaseException:
                await self.aclose()
                raise

    return ProposalScoringClient(
        tokenizer, encoder, max_prefix_tokens=2048, scoring_mode="requested",
        timeout=180.0,
    )
