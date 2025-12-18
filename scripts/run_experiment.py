"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json

from trustquerynet.active.loop import run_active_learning
from trustquerynet.config.schema import load_config
from trustquerynet.training.trainer import train_one_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TrustQueryNet experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("active_learning", {}).get("enabled", False):
        report = run_active_learning(cfg)
        print(json.dumps({"output_dir": report.output_dir, "final_metrics": report.final_metrics}, indent=2))
        return

    artifacts = train_one_run(cfg)
    print(json.dumps({"output_dir": artifacts.output_dir, "metrics": artifacts.metrics}, indent=2))


if __name__ == "__main__":
    main()
