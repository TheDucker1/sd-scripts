"""SVD checkpoint import and continuation using tiny random Anima models only."""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from library.anima_lora_compression import compress_snapref_checkpoint, snapref_metadata
from library.anima_models import Anima
from library.anima_reference import AnimaReference, BranchLoRALinear, install_reference_lora


ROOT = Path(__file__).resolve().parents[1]


class TestSvdInitialization(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(35)
        self.config = json.loads((ROOT / "configs/anima_style_toy.json").read_text())
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.output = self.root / "source.safetensors", self.root / "rank32.safetensors"
        base = Anima(**self.config["anima"])
        source_options = dict(ref=dict(rank=128, alpha=128), tgt=dict(rank=64, alpha=64))
        install_reference_lora(base, **source_options)
        model = AnimaReference(base)
        weights = {}
        for name, module in model.named_modules():
            if not isinstance(module, BranchLoRALinear):
                continue
            for branch, options in module.branches.items():
                prefix = "ref" if branch == "ref" else "lora"
                root = "anima_ref_kv." + branch + "." + name.removeprefix("model.")
                for leaf, direction in (("A", "lora_down"), ("B", "lora_up")):
                    parameter = getattr(module, prefix + "_" + leaf)
                    weights[root + "." + direction] = (torch.randn_like(parameter) * 0.02).bfloat16()
                weights[root + ".alpha"] = torch.tensor(float(options["alpha"]))
        self.modes = torch.stack([torch.linspace(-0.1, 0.1, 64), torch.linspace(0.2, 0.3, 64)]).bfloat16()
        weights["anima_ref_kv.mode_embed"] = self.modes
        save_file(weights, str(self.source), metadata={"anima_ref_kv": json.dumps(dict(
            format="anima_ref_kv/1", ref_t=10, modes=["cs", "char"], lora=source_options, step=123))})

    def compress(self):
        return compress_snapref_checkpoint(self.source, self.output, 32, 32)

    def test_output_preserves_modes_metadata_and_fp32_storage(self):
        receipt, reused = self.compress()
        self.assertFalse(reused)
        self.assertEqual(len(receipt["modules"]), 24)
        _, _, metadata = snapref_metadata(self.output)
        self.assertEqual(metadata["modes"], ["cs", "char"])
        self.assertEqual(metadata["step"], 123)
        self.assertEqual(metadata["lora"], dict(ref=dict(rank=32, alpha=32), tgt=dict(rank=32, alpha=32)))
        with safe_open(str(self.output), framework="pt", device="cpu") as handle:
            torch.testing.assert_close(handle.get_tensor("anima_ref_kv.mode_embed"), self.modes.float(), atol=0, rtol=0)
            self.assertTrue(all(handle.get_slice(key).get_dtype() == "F32" for key in handle.keys()))

    def test_reuse_checks_identity_and_payload_without_rewriting(self):
        receipt, _ = self.compress()
        modified = self.output.stat().st_mtime_ns
        reused_receipt, reused = self.compress()
        self.assertTrue(reused)
        self.assertEqual(receipt, reused_receipt)
        self.assertEqual(self.output.stat().st_mtime_ns, modified)
        with self.assertRaises(FileExistsError):
            compress_snapref_checkpoint(self.source, self.output, 16, 32)
        with self.output.open("r+b") as handle:
            handle.seek(-1, 2)
            previous = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([previous[0] ^ 1]))
        with self.assertRaises(FileExistsError):
            self.compress()

    def test_loaded_rank32_cpu_and_cuda_updates_and_checkpoint_resume(self):
        from anima_train_style import AnimaStyleTrainer
        from library.anima_style_train_utils import build_anima_style_components
        from library.moss_image_style_train_utils import load_style_checkpoint, save_style_checkpoint

        self.compress()
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                config = copy.deepcopy(self.config)
                config.update(adapter=dict(ref=dict(rank=32, alpha=32), tgt=dict(rank=32, alpha=32)),
                              reference_adapter=dict(format="snapref_v1a", path=str(self.output), mode="cs"),
                              dtype="bfloat16" if device == "cuda" else "float32", checkpoint_blocks=2,
                              checkpoint_cpu_offload=device == "cuda", offload_text_encoder=True,
                              attn_mode="flash" if device == "cuda" else "torch")
                config["training"].update(optimizer_type="AdamW8bit" if device == "cuda" else "AdamW", learning_rate=3e-5)
                components = build_anima_style_components(config, torch.device(device))
                model = components.dit
                self.assertEqual(len(model.layers), 2)
                self.assertEqual(tuple(getattr(model, "reference_indices", range(2))), (0, 1))
                self.assertEqual(getattr(model, "reference_group_size", 1), 1)
                self.assertTrue(getattr(model, "reference_cross_attention", True))
                torch.testing.assert_close(model.reference_mode.detach().cpu(), self.modes[0].float(), atol=0, rtol=0)
                self.assertTrue(all(module.branches[branch]["rank"] == 32 for module in model.modules()
                                    if isinstance(module, BranchLoRALinear) for branch in module.branches))
                trainer = AnimaStyleTrainer()
                optimizer = trainer.get_optimizer(config, [p for p in model.parameters() if p.requires_grad])
                before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
                for _ in range(2):
                    clean = torch.randn(1, 16, 4, 4, device=device)
                    reference, noise = torch.randn_like(clean), torch.randn_like(clean)
                    context = torch.randn(1, 3, 32, device=device, dtype=model.dtype)
                    prediction = model((0.05 * clean + 0.95 * noise).to(model.dtype),
                                       torch.tensor([950.0], device=device), context,
                                       reference.to(model.dtype), ref_context=context)
                    loss = torch.nn.functional.mse_loss(prediction.float(), noise - clean)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(all(p.dtype == p.grad.dtype == torch.float32 for p in model.parameters()
                                        if p.requires_grad and p.grad is not None))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for suffix in ("ref_B", "lora_B"):
                    self.assertTrue(any(name.endswith(suffix) and not torch.equal(p, before[name])
                                        for name, p in model.named_parameters() if p.requires_grad))
                self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
                expected = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
                folder = self.root / device
                save_style_checkpoint(folder, components, optimizer, config, 2, 2, random.Random(42))
                with torch.no_grad():
                    for p in model.parameters():
                        if p.requires_grad:
                            p.zero_()
                restored = load_style_checkpoint(folder, components, optimizer, config)
                self.assertEqual(restored["step"], 2)
                self.assertEqual(restored["batches_seen"], 2)
                self.assertTrue(optimizer.state)
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p, expected[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
