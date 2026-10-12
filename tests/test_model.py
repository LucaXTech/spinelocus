import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Install training extras for numerical model tests")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(2)
        self.cfg = {
            "model_name": "smoke_cnn",
            "crop_size": 32,
            "heatmap_size": 16,
            "sigma": 1.5,
            "temperature_ce": 0.5,
            "temperature_decode": 0.35,
        }

    def test_loss_backward_is_finite(self):
        import torch

        from spinelocus.model import HRNetHeatmap, loss

        model = HRNetHeatmap(self.cfg)
        xy = torch.tensor([[7.5, 15.5], [16.0, 4.0]])
        value = loss(model(torch.randn(2, 3, 32, 32)), xy, self.cfg)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_decode_and_flip_share_pixel_center_convention(self):
        import torch

        from spinelocus.model import decode

        logits = torch.full((1, 1, 16, 16), -100.0)
        logits[0, 0, 3, 5] = 100.0
        a = decode(logits, 32).numpy()
        b = decode(logits.flip(-1), 32).numpy()
        np.testing.assert_allclose(a, [[10.5, 6.5]])
        np.testing.assert_allclose(b[:, 0], 31 - a[:, 0])

    def test_yolo_selection_does_not_need_gt(self):
        from types import SimpleNamespace

        import torch

        from spinelocus.evaluate import yolo_points

        class Boxes:
            conf = torch.tensor([0.2, 0.8])

            def __len__(self):
                return 2

        result = SimpleNamespace(
            boxes=Boxes(),
            keypoints=SimpleNamespace(
                xy=torch.stack([torch.zeros(11, 2), torch.full((11, 2), 12.0)]), conf=torch.ones(2, 11)
            ),
        )
        np.testing.assert_allclose(yolo_points(result), np.full((11, 2), 12.0))

    @unittest.skipUnless(importlib.util.find_spec("timm") is not None, "timm not installed")
    def test_real_hrnet_reference_and_high_resolution_forward(self):
        import torch

        from spinelocus.model import HRNetHeatmap

        for index in (1, 3):
            cfg = {
                **self.cfg,
                "model_name": "hrnet_w18",
                "feature_index": index,
                "crop_size": 160,
                "heatmap_size": 64,
            }
            model = HRNetHeatmap(cfg, pretrained=False).eval()
            with torch.inference_mode():
                logits = model(torch.zeros(1, 3, 160, 160))
            self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
            self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipUnless(importlib.util.find_spec("timm") is not None, "timm not installed")
    def test_candidate_backpropagates_through_hrnet_stage4(self):
        import torch

        from spinelocus.model import HRNetHeatmap, loss

        cfg = {**self.cfg, "model_name": "hrnet_w18", "feature_index": 1}
        model = HRNetHeatmap(cfg).eval()
        value = loss(model(torch.randn(2, 3, 64, 64)), torch.tensor([[8., 12.], [20., 10.]]), cfg)
        value.backward()
        grads = [p.grad for name, p in model.backbone.named_parameters() if name.startswith("stage4.")]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))


if __name__ == "__main__":
    unittest.main()
