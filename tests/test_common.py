"""
Tests for nanochat.common module - shared utilities.

Run: python -m pytest tests/test_common.py -v
"""

import os
import logging
import pytest
import torch
from unittest.mock import patch, MagicMock

from nanochat.common import (
    _DTYPE_MAP,
    _detect_compute_dtype,
    ColoredFormatter,
    get_base_dir,
    print0,
    is_ddp_requested,
    get_dist_info,
    autodetect_device_type,
    DummyWandb,
    get_peak_flops,
)


class TestDetectComputeDtype:
    """Test compute dtype auto-detection logic."""

    def test_dtype_map_keys(self):
        assert "bfloat16" in _DTYPE_MAP
        assert "float16" in _DTYPE_MAP
        assert "float32" in _DTYPE_MAP

    def test_dtype_map_values(self):
        assert _DTYPE_MAP["bfloat16"] == torch.bfloat16
        assert _DTYPE_MAP["float16"] == torch.float16
        assert _DTYPE_MAP["float32"] == torch.float32

    def test_env_override_bfloat16(self):
        with patch.dict(os.environ, {"NANOCHAT_DTYPE": "bfloat16"}):
            dtype, reason = _detect_compute_dtype()
            assert dtype == torch.bfloat16
            assert "NANOCHAT_DTYPE" in reason

    def test_env_override_float32(self):
        with patch.dict(os.environ, {"NANOCHAT_DTYPE": "float32"}):
            dtype, reason = _detect_compute_dtype()
            assert dtype == torch.float32
            assert "NANOCHAT_DTYPE" in reason

    def test_no_cuda_returns_float32(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove NANOCHAT_DTYPE if set
            env = os.environ.copy()
            env.pop("NANOCHAT_DTYPE", None)
            with patch.dict(os.environ, env, clear=True):
                with patch("torch.cuda.is_available", return_value=False):
                    dtype, reason = _detect_compute_dtype()
                    assert dtype == torch.float32
                    assert "no CUDA" in reason


class TestGetBaseDir:
    """Test get_base_dir function."""

    def test_returns_string(self):
        result = get_base_dir()
        assert isinstance(result, str)

    def test_directory_exists(self):
        result = get_base_dir()
        assert os.path.isdir(result)

    def test_env_override(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = os.path.join(tmpdir, "custom_nanochat")
            with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": custom_dir}):
                result = get_base_dir()
                assert result == custom_dir
                assert os.path.isdir(custom_dir)

    def test_default_in_cache(self):
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("NANOCHAT_BASE_DIR", None)
            with patch.dict(os.environ, env, clear=True):
                result = get_base_dir()
                assert ".cache" in result or "nanochat" in result


class TestColoredFormatter:
    """Test the ColoredFormatter class."""

    def test_format_info(self):
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="Test message",
            args=None, exc_info=None
        )
        output = formatter.format(record)
        assert "Test message" in output

    def test_format_error(self):
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="", lineno=0, msg="Error occurred",
            args=None, exc_info=None
        )
        output = formatter.format(record)
        assert "Error occurred" in output

    def test_colors_dict(self):
        formatter = ColoredFormatter('%(message)s')
        assert "INFO" in formatter.COLORS
        assert "ERROR" in formatter.COLORS
        assert "WARNING" in formatter.COLORS
        assert "DEBUG" in formatter.COLORS
        assert "CRITICAL" in formatter.COLORS


class TestPrint0:
    """Test the print0 function (only prints on rank 0)."""

    def test_prints_on_rank_0(self, capsys):
        with patch.dict(os.environ, {"RANK": "0"}):
            print0("hello from rank 0")
            captured = capsys.readouterr()
            assert "hello from rank 0" in captured.out

    def test_silent_on_other_ranks(self, capsys):
        with patch.dict(os.environ, {"RANK": "1"}):
            print0("should not print")
            captured = capsys.readouterr()
            assert captured.out == ""

    def test_default_rank_is_zero(self, capsys):
        env = os.environ.copy()
        env.pop("RANK", None)
        with patch.dict(os.environ, env, clear=True):
            print0("default rank")
            captured = capsys.readouterr()
            assert "default rank" in captured.out


class TestDDPHelpers:
    """Test DDP-related helper functions."""

    def test_is_ddp_requested_false(self):
        env = os.environ.copy()
        env.pop("RANK", None)
        env.pop("LOCAL_RANK", None)
        env.pop("WORLD_SIZE", None)
        with patch.dict(os.environ, env, clear=True):
            assert is_ddp_requested() is False

    def test_is_ddp_requested_true(self):
        with patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}):
            assert is_ddp_requested() is True

    def test_get_dist_info_no_ddp(self):
        env = os.environ.copy()
        env.pop("RANK", None)
        env.pop("LOCAL_RANK", None)
        env.pop("WORLD_SIZE", None)
        with patch.dict(os.environ, env, clear=True):
            ddp, rank, local_rank, world_size = get_dist_info()
            assert ddp is False
            assert rank == 0
            assert local_rank == 0
            assert world_size == 1


class TestAutodetectDeviceType:
    """Test autodetect_device_type function."""

    def test_returns_cpu_when_no_gpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                result = autodetect_device_type()
                assert result == "cpu"

    def test_returns_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            result = autodetect_device_type()
            assert result == "cuda"


class TestDummyWandb:
    """Test the DummyWandb mock class."""

    def test_log_does_nothing(self):
        wandb = DummyWandb()
        wandb.log({"loss": 0.5}, step=1)  # should not raise

    def test_finish_does_nothing(self):
        wandb = DummyWandb()
        wandb.finish()  # should not raise


class TestGetPeakFlops:
    """Test GPU peak FLOPS lookup table."""

    def test_h100(self):
        flops = get_peak_flops("NVIDIA H100 80GB HBM3")
        assert flops == 989e12

    def test_h100_pcie(self):
        flops = get_peak_flops("NVIDIA H100 PCIe")
        assert flops == 756e12

    def test_a100(self):
        flops = get_peak_flops("NVIDIA A100-SXM4-80GB")
        assert flops == 312e12

    def test_4090(self):
        flops = get_peak_flops("NVIDIA GeForce RTX 4090")
        assert flops == 165.2e12

    def test_unknown_gpu_returns_inf(self):
        flops = get_peak_flops("Unknown GPU Model XYZ")
        assert flops == float('inf')

    def test_case_insensitive(self):
        flops = get_peak_flops("nvidia h100 80gb hbm3")
        assert flops == 989e12

    def test_mi300x(self):
        flops = get_peak_flops("AMD Instinct MI300X")
        assert flops == 1.3074e15
