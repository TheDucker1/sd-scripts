"""Small CPU fixtures for adapter-only SnapRef exports."""

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from tools.export_anima_snapref import export_adapter, read_checkpoint


class TestExportAnimaSnapRef(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        module = "model.blocks.0.self_attn.q_proj"
        self.config = dict(
            format="anima-reference-lora", format_version=1,
            metadata=dict(model_config=dict(preset="anima-base-v1.0", num_blocks=28)),
            modules={module: dict(in_features=2, out_features=3,
                                 branches=dict(ref=dict(rank=2, alpha=2), tgt=dict(rank=1, alpha=1)))},
        )
        self.tensors = {
            "reference_mode": torch.zeros(2048),
            module + ".ref_A": torch.full((2, 2), 0.125),
            module + ".ref_B": torch.full((3, 2), 0.25),
            module + ".lora_A": torch.full((1, 2), 0.5),
            module + ".lora_B": torch.full((3, 1), 0.75),
        }
        self.source = self.root / "adapter_model.safetensors"
        save_file(self.tensors, str(self.source), metadata={"adapter_config": json.dumps(self.config)})
        self.staging = self.root / "staging"
        self.staging.mkdir()

    def test_adapter_only_inputs_preserve_separate_streams(self):
        archive = self.root / "checkpoint.zip"
        with ZipFile(archive, "w") as handle:
            handle.write(self.source, "step-012000/adapter_model.safetensors")
        for path in (self.root, self.source, archive):
            with self.subTest(input=path):
                config, weights, step = read_checkpoint(path, self.staging)
                self.assertIsNone(step)
                output = self.root / "comfyui.safetensors"
                export_adapter(config, weights, 12000, output)
                with safe_open(str(output), framework="pt", device="cpu") as handle:
                    metadata = json.loads(handle.metadata()["anima_ref_kv"])
                    self.assertEqual(metadata["step"], 12000)
                    self.assertEqual(metadata["modes"], ["cs"])
                    self.assertEqual(len(handle.keys()), 7)
                    self.assertTrue(torch.equal(handle.get_tensor("anima_ref_kv.mode_embed"), torch.zeros(1, 2048).bfloat16()))
                    for branch, rank, down_value, up_value in (("ref", 2, 0.125, 0.25), ("tgt", 1, 0.5, 0.75)):
                        key = f"anima_ref_kv.{branch}.blocks.0.self_attn.q_proj"
                        down = torch.full((rank, 2), down_value).bfloat16()
                        up = torch.full((3, rank), up_value).bfloat16()
                        self.assertTrue(torch.equal(handle.get_tensor(key + ".lora_down"), down))
                        self.assertTrue(torch.equal(handle.get_tensor(key + ".lora_up"), up))
                        self.assertEqual(handle.get_tensor(key + ".alpha").item(), rank)

    def test_full_checkpoint_reads_sidecar_and_step(self):
        (self.root / "adapter_config.json").write_text(json.dumps(self.config))
        (self.root / "complete.json").write_text('{"step": 42}')
        config, weights, step = read_checkpoint(self.root, self.staging)
        self.assertEqual(config, self.config)
        self.assertEqual(weights, self.source)
        self.assertEqual(step, 42)

    def test_missing_configuration_has_clear_error(self):
        save_file(self.tensors, str(self.source))
        with self.assertRaisesRegex(ValueError, "Missing adapter_config.json and embedded"):
            read_checkpoint(self.source, self.staging)


if __name__ == "__main__":
    unittest.main()
