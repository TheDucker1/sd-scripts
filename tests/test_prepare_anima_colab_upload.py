"""Package tiny adapter files while preserving the weight-only initialization."""

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from library.danbooru_index import file_sha256, write_index_json
from tools.prepare_anima_colab_upload import prepare_upload
from tools.colab.stage_anima_checkpoint import stage_checkpoint


class TestPrepareAnimaColabUpload(unittest.TestCase):
    def test_bundle_reuses_adapter_and_excludes_source_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / "step-020000", root / "upload"
            checkpoint.mkdir()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"tiny adapter fixture")
            write_index_json(checkpoint / "adapter_config.json", dict(format="anima-reference-lora", format_version=1,
                weights_sha256=file_sha256(checkpoint / "adapter_model.safetensors")))
            write_index_json(checkpoint / "complete.json", dict(step=20000))
            (checkpoint / "training_state.pt").write_bytes(b"old optimizer state")

            prepare_upload(checkpoint, output)
            previous = (output / "init-step-020000.zip").stat().st_mtime_ns
            prepare_upload(checkpoint, output)
            self.assertEqual((output / "init-step-020000.zip").stat().st_mtime_ns, previous)
            with ZipFile(output / "init-step-020000.zip") as archive:
                self.assertEqual(len(archive.namelist()), 2)
            staged = stage_checkpoint(output / "init-step-020000.zip", root / "ssd")
            self.assertEqual((staged / "adapter_model.safetensors").read_bytes(), b"tiny adapter fixture")
            self.assertFalse((staged / "training_state.pt").exists())


if __name__ == "__main__":
    unittest.main()
