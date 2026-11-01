"""Small offline checks for packed spatial grids and HTTP body reconnection."""

import io
import unittest
from unittest import mock
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from library.moss_image_edit import MossImageEdit, inject_edit_lora
from library.moss_image_style_train_utils import FrozenImageFeatures
from library.moss_image_utils import encode_tokens
from library.streaming_dataset import HTTPRangeStream, PrefetchIterator


class TestPackedMoss(unittest.TestCase):
    def test_mixed_grids_match_individual_predictions_and_have_adapter_gradients(self):
        torch.manual_seed(11)
        model = MossImageEdit(dim=32, ffn_dim=64, num_heads=2, num_layers=2,
                              text_dim=24, freq_dim=16, latent_channels=8)
        inject_edit_lora(model, rank=3, alpha=3)
        targets = [torch.randn(8, 2, 3), torch.randn(8, 1, 2)]
        references = [torch.randn(8, 2, 2), torch.randn(8, 3, 1)]
        context = torch.randn(2, 5, 24)
        mask = torch.tensor([[True, True, True, False, False], [True] * 5])
        timesteps = torch.tensor([250.0, 450.0])
        with sdpa_kernel(SDPBackend.MATH):
            with torch.no_grad():
                expected = [model(target[None], timesteps[index:index + 1], context[index:index + 1],
                                  reference[None], mask[index:index + 1])[0]
                            for index, (target, reference) in enumerate(zip(targets, references))]
            actual = model(targets, timesteps, context, references, mask)
            for value, baseline in zip(actual, expected):
                torch.testing.assert_close(value, baseline, rtol=2e-5, atol=2e-5)
            sum(value.square().mean() for value in actual).backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(value is not None and value.abs().sum() > 0 for value in gradients))
        self.assertTrue(all(value is None or torch.isfinite(value).all() for value in gradients))


class Response(io.BytesIO):
    def __init__(self, payload, length, status=200, content_range=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(length)}
        if content_range:
            self.headers["Content-Range"] = content_range


class TestComponentBatching(unittest.TestCase):
    def test_fixed_feature_resolution_batches_mixed_images_with_pixel_gradients(self):
        class ImageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(3, 4), requires_grad=False)
                self.seen = []

            def forward(self, pixel_values, return_dict):
                self.seen.append(tuple(pixel_values.shape))
                pooled = pixel_values.mean(dim=(-2, -1)) @ self.weight
                return SimpleNamespace(pooler_output=pooled)

        model = ImageModel()
        features = FrozenImageFeatures(model, 8, [0.5] * 3, [0.5] * 3, "pooler")
        images = [torch.randn(3, 8, 12, requires_grad=True), torch.randn(3, 12, 8, requires_grad=True)]
        features(images).sum().backward()
        self.assertEqual(model.seen, [(2, 3, 8, 8)])
        self.assertTrue(all(image.grad is not None and torch.isfinite(image.grad).all() for image in images))

    def test_text_length_grouping_restores_order_without_padding(self):
        class TextModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.seen = []

            def forward(self, input_ids, use_cache, return_dict):
                self.seen.append(tuple(input_ids.shape))
                return SimpleNamespace(last_hidden_state=input_ids.float()[..., None])

        model = TextModel()
        tokens = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6, 7])]
        result = encode_tokens(model, tokens, batch_equal_length=True)
        self.assertEqual(model.seen, [(2, 2), (1, 3)])
        for actual, expected in zip(result, tokens):
            torch.testing.assert_close(actual[:, 0], expected.float())


class TestHTTPRangeStream(unittest.TestCase):
    def test_short_body_resumes_without_repeating_bytes(self):
        responses = [Response(b"abcde", 10), Response(b"fghij", 5, 206, "bytes 5-9/10")]
        with mock.patch("library.streaming_dataset.urlopen", side_effect=responses) as open_url:
            with mock.patch("library.streaming_dataset.time.sleep"):
                with HTTPRangeStream("https://example.invalid/archive.tar.gz") as stream:
                    self.assertEqual(stream.read(10), b"abcde")
                    self.assertEqual(stream.read(10), b"fghij")
                    self.assertEqual(stream.read(10), b"")
            self.assertEqual(open_url.call_args_list[1].args[0].get_header("Range"), "bytes=5-")

    def test_resume_cannot_append_an_ignored_range(self):
        responses = [Response(b"abcde", 10), Response(b"abcdefghij", 10)]
        with mock.patch("library.streaming_dataset.urlopen", side_effect=responses):
            with mock.patch("library.streaming_dataset.time.sleep"):
                with HTTPRangeStream("https://example.invalid/archive.tar", retries=1) as stream:
                    self.assertEqual(stream.read(10), b"abcde")
                    with self.assertRaises(OSError):
                        stream.read(10)


class TestPrefetchIterator(unittest.TestCase):
    def test_order_and_exhaustion(self):
        stream = PrefetchIterator(iter(range(8)), capacity=2)
        try:
            self.assertEqual(list(stream), list(range(8)))
            self.assertFalse(stream.worker.is_alive())
        finally:
            stream.close()

    def test_producer_errors_reach_training_thread(self):
        def source():
            yield "first"
            raise OSError("download failed")

        stream = PrefetchIterator(source(), capacity=2)
        try:
            self.assertEqual(next(stream), "first")
            with self.assertRaisesRegex(OSError, "download failed"):
                next(stream)
        finally:
            stream.close()
