"""
Tests for nanochat.report module - training report utilities.

Run: python -m pytest tests/test_report.py -v
"""

import os
import tempfile
import datetime
import pytest
from unittest.mock import patch

from nanochat.report import (
    run_command,
    slugify,
    extract,
    extract_timestamp,
    estimate_cost,
    Report,
)


class TestRunCommand:
    """Test the run_command helper."""

    def test_successful_command(self):
        result = run_command("echo hello")
        assert result == "hello"

    def test_failed_command(self):
        result = run_command("false")
        # returns "" for return code 0 with no output, or None for failure
        # 'false' returns exit code 1 with no stdout
        assert result is None

    def test_timeout_command(self):
        result = run_command("sleep 100")
        # Should return None due to timeout (5s)
        assert result is None

    def test_command_with_output(self):
        result = run_command("echo -n 'test output'")
        assert result == "test output"


class TestSlugify:
    """Test the slugify function."""

    def test_simple(self):
        assert slugify("Hello World") == "hello-world"

    def test_already_lowercase(self):
        assert slugify("hello") == "hello"

    def test_multiple_spaces(self):
        assert slugify("Base Model Training") == "base-model-training"

    def test_single_word(self):
        assert slugify("Training") == "training"


class TestExtract:
    """Test the extract function."""

    def test_single_key(self):
        section = "- Loss: 0.5\n- Accuracy: 0.9\n"
        result = extract(section, "Loss")
        assert result == {"Loss": "0.5"}

    def test_multiple_keys(self):
        section = "- ARC-Easy: 0.75\n- MMLU: 0.60\n- Other: 0.5\n"
        result = extract(section, ["ARC-Easy", "MMLU"])
        assert result == {"ARC-Easy": "0.75", "MMLU": "0.60"}

    def test_key_not_found(self):
        section = "- Loss: 0.5\n"
        result = extract(section, "Missing")
        assert result == {}

    def test_empty_section(self):
        result = extract("", "anything")
        assert result == {}


class TestExtractTimestamp:
    """Test the extract_timestamp function."""

    def test_valid_timestamp(self):
        content = "Run started: 2026-01-15 10:30:00\nOther stuff\n"
        result = extract_timestamp(content, "Run started")
        assert result == datetime.datetime(2026, 1, 15, 10, 30, 0)

    def test_no_match(self):
        content = "Some random content\n"
        result = extract_timestamp(content, "Run started")
        assert result is None

    def test_invalid_format(self):
        content = "Run started: not-a-date\n"
        result = extract_timestamp(content, "Run started")
        assert result is None


class TestEstimateCost:
    """Test the estimate_cost function."""

    def test_no_gpu(self):
        gpu_info = {"available": False}
        result = estimate_cost(gpu_info)
        assert result is None

    def test_h100_cost(self):
        gpu_info = {"available": True, "count": 8, "names": ["NVIDIA H100 80GB HBM3"]}
        result = estimate_cost(gpu_info)
        assert result is not None
        assert result["hourly_rate"] == 3.00 * 8
        assert result["gpu_type"] == "NVIDIA H100 80GB HBM3"

    def test_a100_cost(self):
        gpu_info = {"available": True, "count": 4, "names": ["NVIDIA A100-SXM4-80GB"]}
        result = estimate_cost(gpu_info)
        assert result is not None
        assert result["hourly_rate"] == 1.79 * 4

    def test_unknown_gpu_uses_default(self):
        gpu_info = {"available": True, "count": 2, "names": ["Unknown GPU"]}
        result = estimate_cost(gpu_info)
        assert result is not None
        assert result["hourly_rate"] == 2.0 * 2  # default rate

    def test_with_runtime(self):
        gpu_info = {"available": True, "count": 1, "names": ["NVIDIA H100"]}
        result = estimate_cost(gpu_info, runtime_hours=2.0)
        assert result["estimated_total"] == 3.00 * 2.0

    def test_without_runtime(self):
        gpu_info = {"available": True, "count": 1, "names": ["NVIDIA H100"]}
        result = estimate_cost(gpu_info)
        assert result["estimated_total"] is None


class TestReport:
    """Test the Report class."""

    def test_init_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = os.path.join(tmpdir, "reports")
            report = Report(report_dir)
            assert os.path.isdir(report_dir)

    def test_log_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(tmpdir)
            file_path = report.log("Test Section", [{"key1": "value1", "key2": 42}])
            assert os.path.exists(file_path)
            assert file_path.endswith("test-section.md")

    def test_log_string_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(tmpdir)
            file_path = report.log("My Section", ["Some plain text\n"])
            with open(file_path) as f:
                content = f.read()
            assert "## My Section" in content
            assert "Some plain text" in content
            assert "timestamp:" in content

    def test_log_dict_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(tmpdir)
            file_path = report.log("Metrics", [{"loss": 0.1234, "steps": 50000}])
            with open(file_path) as f:
                content = f.read()
            assert "- loss: 0.1234" in content
            assert "- steps: 50,000" in content  # large int formatting

    def test_log_skips_falsy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(tmpdir)
            file_path = report.log("Section", [None, {}, "valid\n"])
            with open(file_path) as f:
                content = f.read()
            assert "valid" in content

    def test_log_float_formatting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(tmpdir)
            file_path = report.log("Section", [{"pi": 3.14159265}])
            with open(file_path) as f:
                content = f.read()
            assert "- pi: 3.1416" in content  # 4 decimal places

    def test_generate_creates_report_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Report(tmpdir)
            # Log a section first
            report.log("Test Section", [{"metric": "value"}])
            # Generate should create report.md
            report.generate()
            report_path = os.path.join(tmpdir, "report.md")
            assert os.path.exists(report_path)
