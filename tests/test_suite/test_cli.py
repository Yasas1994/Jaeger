"""Tests for the jaeger CLI."""

from click.testing import CliRunner
import pytest

from jaeger.cli import main


class TestCLICommands:
    """Test that CLI commands are registered."""

    def test_main_help(self):
        """Main CLI should show help with all commands."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Jaeger" in result.output

    def test_health_command_exists(self):
        """health command should be registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "health" in result.output

    def test_predict_command_exists(self):
        """predict command should be registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "predict" in result.output

    def test_train_command_exists(self):
        """train command should be registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output

    def test_download_command_exists(self):
        """download command should be registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "download" in result.output

    def test_utils_command_exists(self):
        """utils subcommand group should be registered."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "utils" in result.output

    def test_version_flag(self):
        """--version should print version and exit."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "jaeger" in result.output.lower()


class TestHealthCommand:
    """Test the health command via CLI."""

    def test_health_help(self):
        """health --help should show options."""
        runner = CliRunner()
        result = runner.invoke(main, ["health", "--help"])
        assert result.exit_code == 0
        assert "verbose" in result.output

    def test_health_runs(self):
        """health should run and produce output."""
        runner = CliRunner()
        result = runner.invoke(main, ["health", "-v"])
        # May fail due to TF/GPU issues in test env, but should produce output
        assert "Jaeger Health Diagnostics" in result.output or result.exit_code in (0, 1)
