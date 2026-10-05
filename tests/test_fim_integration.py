"""CLI and checkpoint contracts for reference, V2, and legacy selectors."""
import contextlib
import io
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcounter import RTCounterConfig, engine  # noqa: E402
from tools import train  # noqa: E402


@dataclass
class _StubConfig:
    yoloe_weights: str = "weights/yoloe.pt"
    text_weights: str = "weights/text.pt"
    use_vpt: bool = True
    vpt_type: str = "reference"
    enhancer_type: str = "reference_fim"
    fim_downsample_ratio: int = 2
    downsample_rate: int = 4
    global_ratio: float = 0.25
    local_kernel: int = 3


class _StubModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded_state = None
        self.strict = None
        self.device = None
        self.is_eval = False

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state = state_dict
        self.strict = strict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True
        return self


class TestEnhancerCli(unittest.TestCase):
    def test_reference_fim_and_vpt_are_cli_defaults(self):
        args = train.get_args(["--data_root", "data"])
        self.assertEqual(args.enhancer_type, "reference_fim")
        self.assertEqual(args.vpt_type, "reference")
        self.assertIsNone(args.fim_downsample_ratio)
        self.assertIsNone(args.downsample_rate)
        self.assertIsNone(args.global_ratio)

    def test_backend_specific_flags_are_not_silently_ignored(self):
        historical = train.get_args(["--data_root", "data", "--enhancer_type", "historical_fim"])
        self.assertEqual(historical.fim_downsample_ratio, 2)

        weaformer = train.get_args([
            "--data_root", "data", "--enhancer_type", "weaformer", "--vpt_type", "legacy",
            "--downsample_rate", "1", "--global_ratio", "0.5",
        ])
        self.assertEqual((weaformer.downsample_rate, weaformer.global_ratio), (1, 0.5))
        self.assertIsNone(weaformer.fim_downsample_ratio)

        for argv, expected in (
            (["--data_root", "data", "--downsample_rate", "1"], "--enhancer_type weaformer"),
            (["--data_root", "data", "--fim_downsample_ratio", "2"], "only valid with --enhancer_type historical_fim"),
            (["--data_root", "data", "--enhancer_type", "reference_fim", "--global_ratio", "0.5"],
             "--enhancer_type weaformer"),
            (["--data_root", "data", "--enhancer_type", "historical_fim", "--downsample_rate", "1"],
             "--enhancer_type weaformer"),
            (["--data_root", "data", "--enhancer_type", "historical_fim", "--fim_downsample_ratio", "1"],
             "must be an integer >= 2"),
            (["--data_root", "data", "--enhancer_type", "weaformer", "--fim_downsample_ratio", "2"],
             "only valid with --enhancer_type historical_fim"),
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                train.get_args(argv)
            self.assertIn(expected, stderr.getvalue())

    def test_reference_and_v2_fim_reject_incompatible_vpt_unless_disabled(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            train.get_args(["--data_root", "data", "--vpt_type", "legacy"])
        self.assertIn("reference_fim requires --vpt_type reference", stderr.getvalue())

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            train.get_args(["--data_root", "data", "--enhancer_type", "v2_fim"])
        self.assertIn("v2_fim requires --vpt_type v2", stderr.getvalue())

        v2 = train.get_args([
            "--data_root", "data", "--enhancer_type", "v2_fim", "--vpt_type", "v2",
        ])
        self.assertEqual((v2.enhancer_type, v2.vpt_type), ("v2_fim", "v2"))

        args = train.get_args(["--data_root", "data", "--vpt_type", "legacy", "--no_vpt"])
        self.assertTrue(args.no_vpt)

    def test_config_defaults_and_weaformer_legacy_defaults(self):
        cfg = RTCounterConfig()
        self.assertEqual((cfg.enhancer_type, cfg.vpt_type), ("reference_fim", "reference"))
        weaformer = RTCounterConfig(enhancer_type="weaformer", vpt_type="legacy")
        self.assertEqual((weaformer.downsample_rate, weaformer.global_ratio, weaformer.local_kernel), (4, 0.25, 3))


class TestCheckpointSelection(unittest.TestCase):
    def _write_checkpoint(self, payload):
        temporary_dir = tempfile.TemporaryDirectory()
        path = Path(temporary_dir.name) / "checkpoint.pt"
        torch.save(payload, path)
        self.addCleanup(temporary_dir.cleanup)
        return path

    def _load_with_stubs(self, path, **overrides):
        with patch.object(engine, "RTCounterConfig", _StubConfig), patch.object(
            engine, "RTCounter", _StubModel
        ):
            return engine.load_model(path, "cpu", **overrides)

    def test_legacy_weaformer_and_vpt_layouts_are_inferred_and_strictly_loaded(self):
        state_dict = {
            "enhancer.weaformers.0.norm1.weight": torch.ones(1),
            "vpt.prototype_queries": torch.ones(1),
        }
        path = self._write_checkpoint({"config": {"text_weights": "old.pt", "use_vpt": True}, "model": state_dict})

        model, _ = self._load_with_stubs(path)

        self.assertEqual((model.cfg.enhancer_type, model.cfg.vpt_type), ("weaformer", "legacy"))
        self.assertIsNone(model.cfg.yoloe_weights)
        self.assertIsNotNone(model.loaded_state)
        self.assertTrue(model.strict)
        self.assertTrue(model.is_eval)

    def test_reference_fim_and_vpt_layouts_are_inferred_before_overlapping_layouts(self):
        state_dict = {
            "enhancer.fim_blocks.0.norm_x.weight": torch.ones(1),
            "vpt.prototype_queries": torch.ones(1),
            "vpt.prototype_extraction.wq.weight": torch.ones(1),
        }
        path = self._write_checkpoint({"config": {"use_vpt": True}, "model": state_dict})

        model, _ = self._load_with_stubs(path)

        self.assertEqual((model.cfg.enhancer_type, model.cfg.vpt_type), ("reference_fim", "reference"))
        self.assertTrue(model.strict)

    def test_v2_fim_and_vpt_layouts_are_inferred(self):
        state_dict = {
            "enhancer.fim_blocks.0.norm0.weight": torch.ones(1),
            "vpt.visual_prototype_extractor.prototype_queries": torch.ones(1),
        }
        path = self._write_checkpoint({"config": {"use_vpt": True}, "model": state_dict})

        model, _ = self._load_with_stubs(path)

        self.assertEqual((model.cfg.enhancer_type, model.cfg.vpt_type), ("v2_fim", "v2"))
        self.assertTrue(model.strict)

    def test_historical_fim_and_legacy_vpt_layouts_are_inferred(self):
        state_dict = {
            "enhancer.blocks.0.norm_x.weight": torch.ones(1),
            "vpt.fusion_attn.q.weight": torch.ones(1),
        }
        path = self._write_checkpoint({"config": {"use_vpt": True}, "model": state_dict})

        model, _ = self._load_with_stubs(path)

        self.assertEqual((model.cfg.enhancer_type, model.cfg.vpt_type), ("historical_fim", "legacy"))
        self.assertTrue(model.strict)

    def test_no_vpt_checkpoint_needs_no_vpt_type_inference(self):
        state_dict = {"enhancer.fim_blocks.0.norm0.weight": torch.ones(1)}
        path = self._write_checkpoint({"config": {"use_vpt": False}, "model": state_dict})

        model, _ = self._load_with_stubs(path)

        self.assertFalse(model.cfg.use_vpt)
        self.assertEqual(model.cfg.enhancer_type, "v2_fim")
        self.assertTrue(model.strict)

    def test_explicit_checkpoint_selectors_are_respected(self):
        state_dict = {
            "enhancer.fim_blocks.0.norm_x.weight": torch.ones(1),
            "vpt.prototype_extraction.wq.weight": torch.ones(1),
        }
        path = self._write_checkpoint({
            "config": {"enhancer_type": "v2_fim", "vpt_type": "v2", "use_vpt": True},
            "model": state_dict,
        })

        model, _ = self._load_with_stubs(path)

        self.assertEqual((model.cfg.enhancer_type, model.cfg.vpt_type), ("v2_fim", "v2"))
        self.assertTrue(model.strict)

    def test_unknown_layouts_require_explicit_overrides(self):
        path = self._write_checkpoint({"config": {"use_vpt": False}, "model": {"enhancer.unknown.weight": torch.ones(1)}})

        with self.assertRaisesRegex(ValueError, "Refusing to guess enhancer compatibility"):
            self._load_with_stubs(path)

        model, _ = self._load_with_stubs(path, enhancer_type="historical_fim")
        self.assertEqual(model.cfg.enhancer_type, "historical_fim")
        self.assertTrue(model.strict)

    def test_unknown_vpt_layout_requires_an_explicit_override(self):
        state_dict = {
            "enhancer.fim_blocks.0.norm0.weight": torch.ones(1),
            "vpt.unrecognised.weight": torch.ones(1),
        }
        path = self._write_checkpoint({"config": {"enhancer_type": "v2_fim", "use_vpt": True}, "model": state_dict})

        with self.assertRaisesRegex(ValueError, "Refusing to guess VPT compatibility"):
            self._load_with_stubs(path)

        model, _ = self._load_with_stubs(path, vpt_type="v2")
        self.assertEqual(model.cfg.vpt_type, "v2")
        self.assertTrue(model.strict)


if __name__ == "__main__":
    unittest.main()
