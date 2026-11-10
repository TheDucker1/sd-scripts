"""Anima adapter imports using tiny synthetic checkpoints only."""

import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from safetensors.torch import save_file
import torch

from library.anima_models import Anima
from library.anima_reference import AnimaReference, BranchLoRALinear, install_reference_lora, load_snapref_adapter


class TestSnapRefLoad(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/anima_style_toy.json").read_text())
        model = Anima(**self.config["anima"])
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "adaln_modulation" in name and torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=0.02)
        install_reference_lora(model, **self.config["adapter"])
        self.model = AnimaReference(model)
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "snapref-tiny.safetensors"
        self.weights, self.expected = {}, {}
        for name, module in self.model.named_modules():
            if not isinstance(module, BranchLoRALinear):
                continue
            for branch, options in module.branches.items():
                root = "anima_ref_kv." + branch + "." + name.removeprefix("model.")
                prefix = "ref" if branch == "ref" else "lora"
                for leaf, direction in (("A", "lora_down"), ("B", "lora_up")):
                    parameter = getattr(module, prefix + "_" + leaf)
                    value = (torch.randn_like(parameter) * 0.03).to(torch.bfloat16)
                    self.weights[root + "." + direction] = value
                    self.expected[name + "." + prefix + "_" + leaf] = value.float()
                self.weights[root + ".alpha"] = torch.tensor(float(options["alpha"]))
        self.weights["anima_ref_kv.mode_embed"] = torch.stack((torch.linspace(-0.1, 0.1, 64), torch.ones(64) * 0.2))
        self.metadata = {"anima_ref_kv": json.dumps(dict(format="anima_ref_kv/1", ref_t=10, modes=["cs", "char"],
                                                       step=10000, name="SnapRef", version="tiny-fixture"))}
        self.write()

    def write(self, weights=None):
        save_file(self.weights if weights is None else weights, str(self.path), metadata=self.metadata)

    def assert_imported(self, model, mode=0):
        for name, value in model.named_parameters():
            if name in self.expected:
                torch.testing.assert_close(value.detach().float().cpu(), self.expected[name].to(value.dtype).float())
        torch.testing.assert_close(model.reference_mode.detach().float().cpu(),
                                   self.weights["anima_ref_kv.mode_embed"][mode].to(model.reference_mode.dtype).float())

    def test_imports_every_branch_and_cs_mode_without_changing_base(self):
        base = {name: value.detach().clone() for name, value in self.model.named_parameters() if not value.requires_grad}
        provenance = load_snapref_adapter(self.model, self.path)
        self.assert_imported(self.model)
        self.assertEqual(provenance["mode"], "cs")
        self.assertEqual(provenance["source_step"], 10000)
        self.assertEqual(provenance["tensors"], len(self.weights))
        for name, value in self.model.named_parameters():
            if name in base:
                torch.testing.assert_close(value, base[name])
        self.assertTrue(all(value.requires_grad for name, value in self.model.named_parameters() if name in self.expected))

    def test_char_mode_selects_its_own_learned_embedding(self):
        load_snapref_adapter(self.model, self.path, mode="char")
        self.assert_imported(self.model, mode=1)
        with self.assertRaisesRegex(ValueError, "requested mode"):
            load_snapref_adapter(self.model, self.path, mode="missing")

    def test_invalid_coverage_rank_alpha_or_nonfinite_fails_before_mutating(self):
        key = next(key for key in self.weights if key.endswith(".lora_down"))
        alpha = next(key for key in self.weights if key.endswith(".alpha"))
        original = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        for problem in ("missing", "rank", "alpha", "nonfinite"):
            with self.subTest(problem=problem):
                weights = dict(self.weights)
                if problem == "missing":
                    del weights[key]
                elif problem == "rank":
                    weights[key] = weights[key][:-1]
                elif problem == "alpha":
                    weights[alpha] = torch.tensor(123.0)
                else:
                    weights[key] = torch.full_like(weights[key], float("nan"))
                self.write(weights)
                with self.assertRaises(ValueError):
                    load_snapref_adapter(self.model, self.path)
                for name, value in self.model.named_parameters():
                    torch.testing.assert_close(value, original[name])

    def test_component_builder_warm_starts_arbitrary_tiny_architecture(self):
        from library.anima_style_train_utils import build_anima_style_components

        config = copy.deepcopy(self.config)
        config["reference_adapter"] = dict(format="snapref_v1a", path=str(self.path), mode="cs")
        components = build_anima_style_components(config, torch.device("cpu"))
        self.assert_imported(components.dit)
        self.assertEqual(config["adapter_initialization"]["mode"], "cs")
        self.assertIsNone(components.style)

    def test_bf16_builder_retains_fp32_trainable_storage(self):
        from library.anima_style_train_utils import build_anima_style_components

        config = copy.deepcopy(self.config)
        config["dtype"] = "bfloat16"
        config["reference_adapter"] = dict(format="snapref_v1a", path=str(self.path), mode="cs")
        model = build_anima_style_components(config, torch.device("cpu")).dit
        self.assert_imported(model)
        for parameter in model.parameters():
            self.assertEqual(parameter.dtype, torch.float32 if parameter.requires_grad else torch.bfloat16)

    def test_native_checkpoint_initializes_weights_without_training_state(self):
        from library.anima_style_train_utils import build_anima_style_components
        from library.moss_image_style_train_utils import save_style_checkpoint

        load_snapref_adapter(self.model, self.path, mode="char")
        parameters = [value for value in self.model.parameters() if value.requires_grad]
        checkpoint = Path(self.temporary.name) / "step-000100"
        save_style_checkpoint(checkpoint, SimpleNamespace(dit=self.model), torch.optim.SGD(parameters, lr=1e-4),
                              self.config, 100, 200, random.Random(17))
        (checkpoint / "training_state.pt").unlink()
        (checkpoint / "complete.json").unlink()

        config = copy.deepcopy(self.config)
        config["training"]["max_steps"] = 123
        config["dtype"] = "bfloat16"
        config["reference_adapter"] = dict(format="anima_checkpoint", path=str(checkpoint))
        components = build_anima_style_components(config, torch.device("cpu"))
        self.assert_imported(components.dit, mode=1)
        self.assertEqual(config["adapter_initialization"]["checkpoint"], checkpoint.name)

    @unittest.skipUnless(torch.cuda.is_available(), "Tiny CUDA import check requires a GPU")
    def test_gpu_import_and_optimizer_step_train_both_loaded_streams(self):
        from library.anima_style_train_utils import build_anima_style_components

        config = copy.deepcopy(self.config)
        config["dtype"] = "bfloat16"
        config["reference_adapter"] = dict(format="snapref_v1a", path=str(self.path), mode="cs")
        model = build_anima_style_components(config, torch.device("cuda")).dit
        self.assert_imported(model)
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-4)
        inputs = torch.randn(1, 16, 4, 4, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 3, 32, device="cuda", dtype=torch.bfloat16)
        original = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
        prediction = model(inputs, torch.tensor([500.0], device="cuda"), context, inputs, ref_context=context)
        self.assertEqual(prediction.dtype, torch.bfloat16)
        prediction.float().square().mean().backward()
        self.assertTrue(all(value.dtype == value.grad.dtype == torch.float32
                            for value in model.parameters() if value.requires_grad and value.grad is not None))
        optimizer.step()
        for suffix in ("ref_B", "lora_B"):
            self.assertTrue(any(name.endswith(suffix) and not torch.equal(value, original[name])
                                for name, value in model.named_parameters() if value.requires_grad))
        self.assertTrue(all(value.grad is None for value in model.parameters() if not value.requires_grad))


if __name__ == "__main__":
    unittest.main()
