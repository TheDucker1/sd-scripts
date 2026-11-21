"""Stage only two tiny adapter files, checking corruption and repeat launches."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from tools.colab.stage_anima_checkpoint import stage_checkpoint


class TestStageAnimaCheckpoint(unittest.TestCase):
    def test_stages_weights_only_and_reuses_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"tiny synthetic weights"
            metadata = dict(format="anima-reference-lora", format_version=1, weights_sha256=hashlib.sha256(payload).hexdigest())
            source = root / "init.zip"
            with ZipFile(source, "w") as archive:
                archive.writestr("step-020000/adapter_config.json", json.dumps(metadata))
                archive.writestr("step-020000/adapter_model.safetensors", payload)
                archive.writestr("step-020000/training_state.pt", b"excluded old optimizer")
            target = stage_checkpoint(source, root / "ssd")
            self.assertEqual((target / "adapter_model.safetensors").read_bytes(), payload)
            self.assertFalse((target / "training_state.pt").exists())
            self.assertEqual(stage_checkpoint(source, root / "ssd"), target)

    def test_rejects_corrupt_adapter_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "init.zip"
            with ZipFile(source, "w") as archive:
                archive.writestr("step-020000/adapter_config.json", json.dumps(dict(
                    format="anima-reference-lora", format_version=1, weights_sha256="0" * 64)))
                archive.writestr("step-020000/adapter_model.safetensors", b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                stage_checkpoint(source, root / "ssd")


if __name__ == "__main__":
    unittest.main()
