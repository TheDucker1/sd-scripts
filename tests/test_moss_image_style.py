import copy
import functools
import http.server
import io
import json
import itertools
import os
import random
import tarfile
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from safetensors import safe_open
from torch.utils.data import DataLoader
from torch.nn.attention import SDPBackend, sdpa_kernel

from library.moss_image_edit import MossImageEdit, adapter_state_dict, inject_edit_lora
from library.moss_image_style_dataset import (
    MossStyleBucketBatchDataset, MossStylePairDataset, make_style_buckets, read_image_pairs_batch,
    iter_image_pairs_batch, resolve_style_dataset_config,
)
from library import config_util
from library.dataset import BaseDataset, BucketManager
from library.streaming_dataset import process_stream_caption
from library.moss_image_style_train_utils import (
    build_random_components,
    configure_style_compile,
    prepare_style_batch,
    sample_style_sigmas,
    style_objective,
)
from moss_image_train_style import MossImageStyleTrainer, setup_parser


def make_archive(path, count=8, shapes=None):
    rng = np.random.default_rng(7)
    with tarfile.open(path, "w:gz" if str(path).endswith(".gz") else "w") as archive:
        for index in range(count):
            data = io.BytesIO()
            height, width = shapes[index % len(shapes)] if shapes else (24 + index, 40)
            Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(data, format="PNG")
            payload = data.getvalue()
            member = tarfile.TarInfo(f"original/{index:04d}.png")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        for name, payload in (("broken.png", b"not an image"), ("metadata.json", b"{}")):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def toy_config(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/moss_style_toy.json").read_text())
    archive = tmp_path / "images.tar"
    make_archive(archive)
    config["data"]["shards"] = [str(archive)]
    config["training"].update(max_steps=2, gradient_accumulation_steps=1, sample_every=2, save_every=1)
    return config


def check_stream_pairs_and_corrupt_members(tmp_path, suffix):
    archive = tmp_path / f"images{suffix}"
    make_archive(archive)
    dataset = MossStylePairDataset([archive], resolution=32, repeat=False)
    pairs = list(dataset)
    assert len(pairs) == 4
    ids = [pair[key] for pair in pairs for key in ("id_a", "id_b")]
    assert len(set(ids)) == 8
    assert all(pair["image_a"].shape == (3, 32, 32) for pair in pairs)
    assert all(pair["image_a"].min() >= -1 and pair["image_a"].max() <= 1 for pair in pairs)
    assert [pair["id_a"] for pair in dataset] == [pair["id_a"] for pair in pairs]


def check_http_stream(tmp_path):
    archive = tmp_path / "images.tar"
    make_archive(archive)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(QuietHandler, directory=str(tmp_path))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/images.tar"
            dataset = MossStylePairDataset([url], resolution=16, repeat=False)
            loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=read_image_pairs_batch)
            batches = list(loader)
            assert len(batches) == 2
            assert batches[0]["image_a"].shape == (2, 3, 16, 16)
        finally:
            server.shutdown()
            thread.join()


def check_noise_curve():
    torch.manual_seed(2)
    sigma = sample_style_sigmas(4096, "cpu")
    assert 0.05 < sigma.min() < sigma.max() < 0.95
    assert 0.03 < (sigma > 0.7).float().mean() < 0.07
    assert 0.2 < sigma[sigma <= 0.7].mean() < 0.5


def check_edit_mask_reference_time_and_gradients():
    torch.manual_seed(2)
    model = MossImageEdit(dim=32, ffn_dim=64, num_heads=2, num_layers=2, text_dim=24, freq_dim=16, latent_channels=8)
    inject_edit_lora(model, rank=3, alpha=3)
    x = torch.randn(2, 8, 2, 3)
    ref = torch.randn(2, 8, 3, 2, requires_grad=True)
    text = torch.randn(2, 3, 24)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    # Changing target/text/time cannot change the reference branch activations.
    references = []
    hook = model.layers[-1].register_forward_hook(lambda module, args, output: references.append(output[:, 3:9].detach().clone()))
    first = model(x, torch.tensor([300.0, 400.0]), text, ref, mask)
    second = model(x * 2, torch.tensor([600.0, 650.0]), text + 1, ref, mask)
    hook.remove()
    torch.testing.assert_close(references[0], references[1], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first, second)
    first.square().mean().backward()
    assert ref.grad is not None and ref.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters() if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    changed = model(x, torch.tensor([300.0, 400.0]), text, ref.detach() + 1, mask)
    assert not torch.allclose(first, changed)
    padded = text.clone()
    padded[0, 2] += 100
    torch.testing.assert_close(first[0], model(x, torch.tensor([300.0, 400.0]), padded, ref, mask)[0])


def check_each_loss_reaches_only_adapter(toy_config):
    torch.manual_seed(5)
    components = build_random_components(toy_config, torch.device("cpu"))
    batch = read_image_pairs_batch([next(iter(MossStylePairDataset(**toy_config["data"])))])
    prepared = prepare_style_batch(components, batch, "cpu", random.Random(4))
    assert "watercolor" not in prepared["prompts"][0]
    assert "high resolution" not in prepared["prompts"][0]
    assert next(components.tagger.parameters()).dtype == torch.bfloat16
    noise = torch.randn_like(prepared["latents"])
    for content, style in ((1.0, 0.0), (0.0, 1.0)):
        components.dit.zero_grad(set_to_none=True)
        loss, _ = style_objective(components, prepared, torch.tensor([0.4]), noise, content, style, True, True)
        loss.backward()
        gradients = [parameter.grad for parameter in components.dit.parameters() if parameter.requires_grad]
        assert torch.isfinite(loss) and any(grad is not None and grad.abs().sum() > 0 for grad in gradients)
        assert all(parameter.grad is None for module in components.frozen_modules() for parameter in module.parameters())


def check_training_checkpoint_and_exact_resume(toy_config, tmp_path):
    config = copy.deepcopy(toy_config)
    config["training"].update(gradient_accumulation_steps=2, sample_every=0)
    torch.manual_seed(config["seed"])
    initial = build_random_components(config, torch.device("cpu"))
    before = adapter_state_dict(initial.dit)
    base_before = {
        name: parameter.detach().clone() for name, parameter in initial.dit.named_parameters() if not parameter.requires_grad
    }
    trained, history = MossImageStyleTrainer().train_config(config, tmp_path / "continuous")
    assert len(history) == 2 and all(row["gradient_norm"] > 0 for row in history)
    assert any(not torch.equal(before[name], value) for name, value in adapter_state_dict(trained.dit).items())
    for name, parameter in trained.dit.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(base_before[name], parameter, rtol=0, atol=0)
    exported = load_file(str(tmp_path / "continuous/step-000002/adapter_model.safetensors"))
    assert exported.keys() == before.keys()
    adapter_config = json.loads((tmp_path / "continuous/step-000002/adapter_config.json").read_text())
    assert adapter_config["targets"] == sorted(adapter_config["modules"])
    with safe_open(str(tmp_path / "continuous/step-000002/adapter_model.safetensors"), framework="pt") as handle:
        assert json.loads(handle.metadata()["adapter_config"]) == {
            key: value for key, value in adapter_config.items() if key != "weights_sha256"
        }
    resumed, resumed_history = MossImageStyleTrainer().train_config(
        config, tmp_path / "resumed", resume=tmp_path / "continuous/step-000001"
    )
    for name, value in adapter_state_dict(trained.dit).items():
        torch.testing.assert_close(value, adapter_state_dict(resumed.dit)[name], rtol=0, atol=0)
    assert resumed_history == history[1:]


def check_compiled_forward_backward_matches_eager(toy_config, scope):
    torch.manual_seed(9)
    eager = build_random_components(toy_config, torch.device("cpu"))
    compiled = copy.deepcopy(eager)
    args = setup_parser().parse_args(
        [
            "--config",
            "unused",
            "--output_dir",
            "unused",
            "--compile",
            "--compile_backend",
            "aot_eager",
            "--compile_scope",
            scope,
            "--compile_modules",
            "dit",
            "vae",
            "text",
            "tagger",
            "content",
            "style",
        ]
    )
    configure_style_compile(compiled, args)
    batch = read_image_pairs_batch([next(iter(MossStylePairDataset(**toy_config["data"])))])
    prepared = prepare_style_batch(eager, batch, "cpu", random.Random(4))
    compiled_prepared = prepare_style_batch(compiled, batch, "cpu", random.Random(4))
    noise = torch.randn_like(prepared["latents"])
    losses = []
    for model, inputs in ((eager, prepared), (compiled, compiled_prepared)):
        loss, _ = style_objective(model, inputs, torch.tensor([0.4]), noise, checkpoint_decoder=True, checkpoint_features=True)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], atol=2e-5, rtol=2e-5)
    for (name, a), (other, b) in zip(eager.dit.named_parameters(), compiled.dit.named_parameters()):
        assert name == other
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-5)


def check_toy_training_writes_sample(toy_config, tmp_path):
    toy_config["sampling"].update(num_pairs=2, steps=3, guidance_scale=2.0, flow_shift=5.0)
    expected = list(itertools.islice(MossStylePairDataset(seed=toy_config["seed"], **toy_config["data"]), 4))
    _, history = MossImageStyleTrainer().train_config(toy_config, tmp_path / "sample")
    assert len(history) == 2
    directory = tmp_path / "sample/samples/step-000002"
    for index in range(2):
        path = directory / f"pair-{index:03d}.jpg"
        with Image.open(path) as image:
            assert image.format == "JPEG" and image.size == (96, 56)
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["id_a"] == expected[index + 2]["id_a"]
        assert metadata["id_b"] == expected[index + 2]["id_b"]
        assert metadata["effective_steps"] == 3
        assert metadata["guidance_scale"] == 2.0 and metadata["flow_shift"] == 5.0
        assert metadata["result_size"] == [32, 32]
        assert "watercolor" not in metadata["prompt"]
    state = torch.load(tmp_path / "sample/step-000002/training_state.pt", weights_only=True)
    assert state["batches_seen"] == 4  # Two training batches and two additional review batches.


def check_pixel_buckets(tmp_path):
    for budget in (1_000_000, 768 * 768):
        buckets = make_style_buckets(dict(enable_bucket=True, max_pixels=budget, bucket_reso_steps=64,
                                         min_bucket_reso=256, max_bucket_reso=1536))
        assert len(buckets) > 1
        assert all(width * height <= budget and width % 64 == height % 64 == 0 for width, height in buckets)
        assert any(width < height for width, height in buckets) and any(width > height for width, height in buckets)
    assert make_style_buckets(dict(max_megapixels=1.0)) == make_style_buckets(dict(max_pixels=1_000_000))
    archive = tmp_path / "bucketed.tar"
    make_archive(archive, shapes=[(24, 40), (40, 24), (32, 32), (48, 16)])
    dataset = MossStylePairDataset([archive], max_pixels=1024, bucket_reso_steps=8, min_bucket_reso=16,
                                   max_bucket_reso=64, repeat=False)
    batches = list(MossStyleBucketBatchDataset(dataset, batch_size=2))
    assert sum(len(batch) for batch in batches) == 4
    independent_shapes = False
    for samples in batches:
        batch = read_image_pairs_batch(samples)
        for pair in iter_image_pairs_batch(batch):
            for key in ("image_a", "image_b"):
                height, width = pair[key].shape[-2:]
                assert height * width <= 1024 and height % 8 == width % 8 == 0
            independent_shapes |= pair["image_a"].shape[-2:] != pair["image_b"].shape[-2:]
    assert independent_shapes


def check_fixed_objective_optimizes(config):
    torch.manual_seed(12)
    components = build_random_components(config, torch.device("cpu"))
    batch = read_image_pairs_batch([next(iter(MossStylePairDataset(**config["data"])))])
    prepared = prepare_style_batch(components, batch, "cpu", random.Random(3))
    noise = torch.randn_like(prepared["latents"])
    parameters = [parameter for parameter in components.dit.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = style_objective(components, prepared, torch.tensor([0.4]), noise)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0], losses


class TestMossStyleTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.previous_tempdir = tempfile.tempdir
        cls.previous_environment = {
            name: os.environ.get(name) for name in ("TEMP", "TMP", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
        }
        parent = Path(__file__).resolve().parents[1] / "tests/test_data/moss_style_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(parent)
        for name in cls.previous_environment:
            os.environ[name] = str(parent / name.lower()) if "CACHE" in name else str(parent)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        tempfile.tempdir = cls.previous_tempdir
        for name, value in cls.previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_tar(self):
        check_stream_pairs_and_corrupt_members(self.path, ".tar")

    def test_gzip_tar(self):
        check_stream_pairs_and_corrupt_members(self.path, ".tar.gz")

    def test_http_stream(self):
        check_http_stream(self.path)

    def test_noise(self):
        check_noise_curve()

    def test_reference_contract(self):
        check_edit_mask_reference_time_and_gradients()

    def test_loss_gradients(self):
        check_each_loss_reaches_only_adapter(toy_config(self.path))

    def test_resume(self):
        check_training_checkpoint_and_exact_resume(toy_config(self.path), self.path)

    def test_compile_blocks(self):
        with sdpa_kernel(SDPBackend.MATH):
            check_compiled_forward_backward_matches_eager(toy_config(self.path), "blocks")

    def test_compile_whole(self):
        with sdpa_kernel(SDPBackend.MATH):
            check_compiled_forward_backward_matches_eager(toy_config(self.path), "whole")

    def test_sample(self):
        check_toy_training_writes_sample(toy_config(self.path), self.path)

    def test_pixel_buckets(self):
        check_pixel_buckets(self.path)

    def test_shared_streaming_config_and_no_lookahead(self):
        archive = self.path / "sequential.tar"
        make_archive(archive)
        source = dict(
            general=dict(resolution=32, shuffle_caption=True, keep_tokens=1, caption_prefix="prefix"),
            datasets=[dict(batch_size=2, subsets=[dict(shards=[str(archive)], flip_aug=False)])],
        )
        blueprint = config_util.generate_streaming_blueprint(source)
        assert blueprint.params.resolution == (32, 32) and blueprint.params.batch_size == 2
        subset = blueprint.subsets[0].params
        assert subset.caption_prefix == "prefix" and subset.shuffle_caption and subset.keep_tokens == 1
        pairs = MossStylePairDataset(**source, seed=42)
        assert isinstance(pairs.dataset, BaseDataset) and isinstance(pairs.dataset.bucket_manager, BucketManager)
        with mock.patch.object(pairs.dataset, "_process_image", wraps=pairs.dataset._process_image) as process:
            stream = iter(pairs)
            first = next(stream)
            assert process.call_count == 2
            assert first["id_a"].endswith("0000.png") and first["id_b"].endswith("0001.png")
            assert next(stream)["id_a"].endswith("0002.png") and process.call_count == 4
            stream.close()
        assert pairs.dataset.image_data == {} and pairs.dataset.bucket_manager.resos == []
        caption = process_stream_caption("one, two, three", pairs.dataset.subsets[0], random.Random(3), 0, 10)
        assert "prefix" in caption and "one" in caption
        assert not pairs.dataset.is_latent_cacheable() and not pairs.dataset.is_text_encoder_output_cacheable()

    def test_strict_streaming_rejects_lookahead_options(self):
        config = toy_config(self.path)
        for setting in (dict(shuffle_buffer=2), dict(num_repeats=2), dict(validation_split=0.1)):
            with self.assertRaises(ValueError):
                config_util.generate_streaming_blueprint(dict(datasets=[dict(subsets=[dict(shards=["unused.tar"])], **setting)]))
        config["training"]["num_workers"] = 1
        with self.assertRaises(ValueError):
            resolve_style_dataset_config(config)

    def test_streaming_no_upscale_does_not_register_images(self):
        archive = self.path / "no-upscale.tar"
        make_archive(archive)
        pairs = MossStylePairDataset([archive], resolution=32, enable_bucket=True, bucket_no_upscale=True,
                                     bucket_reso_steps=8, min_bucket_reso=16, max_bucket_reso=64, repeat=False)
        first = next(iter(pairs))
        assert first["image_a"].shape[-2:] == (24, 40)
        assert pairs.dataset.bucket_manager.resos == [] and pairs.dataset.image_data == {}

    def test_fixed_objective_optimizes(self):
        check_fixed_objective_optimizes(toy_config(self.path))

    def test_moss_fullgraph(self):
        model = MossImageEdit(dim=32, ffn_dim=64, num_heads=2, num_layers=1, text_dim=24, freq_dim=16, latent_channels=8)
        inject_edit_lora(model, rank=3, alpha=3)
        model.enable_gradient_checkpointing()
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        with sdpa_kernel(SDPBackend.MATH):
            value = compiled(
                torch.randn(2, 8, 2, 3),
                torch.tensor([300.0, 450.0]),
                torch.randn(2, 3, 24),
                torch.randn(2, 8, 3, 2),
                torch.tensor([[1, 1, 0], [1, 1, 1]]),
            )
            value.square().mean().backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.parameters()
            if parameter.requires_grad
        )
