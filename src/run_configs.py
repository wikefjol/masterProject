import os
from run_scenario import run_scenario


def run_configs(config_dir):
    """
    Run all scenarios in the given configuration directory.
    """
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    
    config_files = [os.path.join(config_dir, f) for f in os.listdir(config_dir) if f.endswith(".json")]
    if not config_files:
        raise ValueError(f"No configuration files found in {config_dir}")

    for config_file in config_files:
        print(f"Running scenario with config: {config_file}")
        run_scenario(config_file)


if __name__ == "__main__":
    import argparse

    # Parse arguments from the command line
    parser = argparse.ArgumentParser(description="Run all scenarios in the given configuration directory.")
    parser.add_argument(
        "config_dir",
        help="Path to the directory containing configuration files."
    )

    args = parser.parse_args()

    # Run the scenarios in the provided directory
    run_configs(args.config_dir)
