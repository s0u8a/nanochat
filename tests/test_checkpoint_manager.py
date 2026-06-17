"""
Tests for nanochat.checkpoint_manager module - checkpoint save/load utilities.

Run: python -m pytest tests/test_checkpoint_manager.py -v
"""

import os
import json
import tempfile
import pytest
import torch
from unittest.mock import patch

from nanochat.checkpoint_manager import (
    _patch_missing_config_keys,
    _patch_missing_keys,
    save_checkpoint,
    load_checkpoint,
    find_largest_model,
    find_last_step,
)


class TestPatchMissingConfigKeys:
    """Test _patch_missing_config_keys for backward compatibility."""

    def test_adds_window_pattern_when_missing(self):
        config = {"n_layer": 12, "n_head": 6}
        _patch_missing_config_keys(config)
        assert config["window_pattern"] == "L"

    def test_preserves_existing_window_pattern(self):
        config = {"n_layer": 12, "window_pattern": "SSSL"}
        _patch_missing_config_keys(config)
        assert config["window_pattern"] == "SSSL"


class TestPatchMissingKeys:
    """Test _patch_missing_keys for backward compatibility."""

    def test_adds_resid_lambdas_when_missing(self):
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            n_layer: int = 4

        model_data = {}
        config = FakeConfig(n_layer=4)
        _patch_missing_keys(model_data, config)
        assert "resid_lambdas" in model_data
        assert model_data["resid_lambdas"].shape == (4,)
        assert torch.all(model_data["resid_lambdas"] == 1.0)

    def test_adds_x0_lambdas_when_missing(self):
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            n_layer: int = 3

        model_data = {}
        config = FakeConfig(n_layer=3)
        _patch_missing_keys(model_data, config)
        assert "x0_lambdas" in model_data
        assert model_data["x0_lambdas"].shape == (3,)
        assert torch.all(model_data["x0_lambdas"] == 0.0)

    def test_preserves_existing_keys(self):
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            n_layer: int = 2

        existing_resid = torch.tensor([0.5, 0.7])
        existing_x0 = torch.tensor([0.1, 0.2])
        model_data = {"resid_lambdas": existing_resid, "x0_lambdas": existing_x0}
        config = FakeConfig(n_layer=2)
        _patch_missing_keys(model_data, config)
        assert torch.equal(model_data["resid_lambdas"], existing_resid)
        assert torch.equal(model_data["x0_lambdas"], existing_x0)


class TestSaveAndLoadCheckpoint:
    """Test save_checkpoint and load_checkpoint round-trip."""

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            step = 100
            model_data = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
            meta_data = {"model_config": {"n_layer": 4}, "step": step}

            save_checkpoint(tmpdir, step, model_data, optimizer_data=None, meta_data=meta_data, rank=0)

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, f"model_{step:06d}.pt"))
            assert os.path.exists(os.path.join(tmpdir, f"meta_{step:06d}.json"))

            # Load and verify
            loaded_model, loaded_optim, loaded_meta = load_checkpoint(tmpdir, step, device="cpu")
            assert torch.allclose(loaded_model["weight"], model_data["weight"])
            assert torch.allclose(loaded_model["bias"], model_data["bias"])
            assert loaded_optim is None
            assert loaded_meta["model_config"]["n_layer"] == 4

    def test_save_load_with_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            step = 50
            model_data = {"w": torch.randn(5, 5)}
            optimizer_data = {"state": torch.randn(5, 5)}
            meta_data = {"step": step}

            save_checkpoint(tmpdir, step, model_data, optimizer_data, meta_data, rank=0)

            # Load with optimizer
            loaded_model, loaded_optim, loaded_meta = load_checkpoint(
                tmpdir, step, device="cpu", load_optimizer=True, rank=0
            )
            assert loaded_optim is not None
            assert torch.allclose(loaded_optim["state"], optimizer_data["state"])

    def test_non_rank0_doesnt_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            step = 10
            model_data = {"w": torch.randn(3, 3)}
            meta_data = {"step": step}

            save_checkpoint(tmpdir, step, model_data, optimizer_data=None, meta_data=meta_data, rank=1)

            # Model and meta should NOT be saved by non-zero rank
            assert not os.path.exists(os.path.join(tmpdir, f"model_{step:06d}.pt"))
            assert not os.path.exists(os.path.join(tmpdir, f"meta_{step:06d}.json"))


class TestFindLargestModel:
    """Test find_largest_model function."""

    def test_finds_largest_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock model directories
            os.makedirs(os.path.join(tmpdir, "d4"))
            os.makedirs(os.path.join(tmpdir, "d12"))
            os.makedirs(os.path.join(tmpdir, "d26"))
            os.makedirs(os.path.join(tmpdir, "d8"))

            result = find_largest_model(tmpdir)
            assert result == "d26"

    def test_single_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "d6"))
            result = find_largest_model(tmpdir)
            assert result == "d6"

    def test_non_standard_names_falls_back_to_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directories with non-standard names
            os.makedirs(os.path.join(tmpdir, "custom_model_a"))
            os.makedirs(os.path.join(tmpdir, "custom_model_b"))

            result = find_largest_model(tmpdir)
            assert result in ["custom_model_a", "custom_model_b"]

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                find_largest_model(tmpdir)


class TestFindLastStep:
    """Test find_last_step function."""

    def test_finds_last_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock checkpoint files
            torch.save({}, os.path.join(tmpdir, "model_000100.pt"))
            torch.save({}, os.path.join(tmpdir, "model_000500.pt"))
            torch.save({}, os.path.join(tmpdir, "model_001000.pt"))

            result = find_last_step(tmpdir)
            assert result == 1000

    def test_single_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save({}, os.path.join(tmpdir, "model_000042.pt"))
            result = find_last_step(tmpdir)
            assert result == 42

    def test_no_checkpoints_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                find_last_step(tmpdir)
