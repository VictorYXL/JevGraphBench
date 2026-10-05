import unittest
from unittest.mock import patch

from src.clients.base import BaseDecisionClient
from src.utils import proposal_scoring_client as scoring


@unittest.skipUnless(scoring.BRIDGE.is_file(), "Requires retained local experiment snapshot")
class ProposalScoringClientTests(unittest.TestCase):
    def test_preserved_bridge_integrity_and_base(self):
        bridge = scoring._load_bridge()
        self.assertTrue(issubclass(bridge.SemIfVLLMBridge, BaseDecisionClient))
        self.assertEqual(bridge.MODEL_ID, "Qwen3.5-4B")
        self.assertEqual(bridge.BASE_URL, "http://127.0.0.1:8002/v1")

    def test_source_drift_refused(self):
        with patch.object(scoring, "BRIDGE_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "source hash differs"):
                scoring._load_bridge()


if __name__ == "__main__":
    unittest.main()
