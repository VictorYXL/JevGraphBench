from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.clients.base import BaseDecisionClient
from src.utils import proposal_scoring_client as scoring


class ProposalScoringClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bridge_path = Path(self.directory.name) / "bridge.py"
        self.bridge_path.write_text(
            "from src.clients.base import BaseDecisionClient\n"
            "class SemIfVLLMBridge(BaseDecisionClient):\n"
            "    pass\n"
            "MODEL_ID = 'Qwen3.5-4B'\n"
            "BASE_URL = 'http://127.0.0.1:8002/v1'\n")
        patcher = patch.object(scoring, "BRIDGE", self.bridge_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        hash_patcher = patch.object(scoring, "BRIDGE_SHA256", sha256(self.bridge_path.read_bytes()).hexdigest())
        hash_patcher.start()
        self.addCleanup(hash_patcher.stop)

    def test_preserved_bridge_integrity_and_base(self):
        bridge = scoring._load_bridge()
        self.assertTrue(issubclass(bridge.SemIfVLLMBridge, BaseDecisionClient))
        self.assertEqual(bridge.MODEL_ID, "Qwen3.5-4B")
        self.assertEqual(bridge.BASE_URL, "http://127.0.0.1:8002/v1")

    def test_source_drift_refused(self):
        with patch.object(scoring, "BRIDGE_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "source hash differs"):
                scoring._load_bridge()

    def test_model_directory_must_be_explicit(self):
        with patch.dict(scoring.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Set GRAPHDECISIONBENCH_SCORING_MODEL"):
                scoring.model_path()
        with patch.dict(scoring.os.environ, {"GRAPHDECISIONBENCH_SCORING_MODEL": self.directory.name}, clear=True):
            self.assertEqual(scoring.model_path(), Path(self.directory.name))
        with patch.dict(scoring.os.environ, {"GRAPHDECISIONBENCH_SCORING_MODEL": str(self.bridge_path)}, clear=True):
            with self.assertRaisesRegex(ValueError, "existing directory"):
                scoring.model_path()

    def test_missing_model_fails_before_loading_external_source(self):
        with patch.dict(scoring.os.environ, {}, clear=True), patch.object(scoring, "_load_bridge") as load:
            with self.assertRaisesRegex(ValueError, "SCORING_MODEL"):
                scoring.create_client()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
