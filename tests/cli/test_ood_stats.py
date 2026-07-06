from click.testing import CliRunner

from src.cli.ood_stats import compute_ood_stats_cmd


def test_compute_ood_stats_cmd_help() -> None:
    result = CliRunner().invoke(compute_ood_stats_cmd, ["--help"])
    assert result.exit_code == 0
    assert "ood_stats" in result.output.lower() or "retraining" in result.output.lower()
