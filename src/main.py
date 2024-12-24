from run_scenario import run_scenario

def main():
    """
    Entry point for running a specific scenario using a configuration file.

    Workflow:
    1. If no configuration files exist in the `scenarios` directory:
       - Locate the `generate_configs.py` script in the `utils` directory.
       - Run it to generate a set of configuration files. These will be saved under the `/scenarios` directory.
    
    2. Once configuration files are generated:
       - Choose one of the generated config files as your scenario.
       - Update the `CONFIG_PATH` variable below with the absolute path to the chosen config file.

    3. Run the script:
       - Execute this script directly using `python3 main.py`.
       - Alternatively, you can directly invoke the `run_scenario.py` script with the config path:
         `python3 run_scenario.py "path_to_config"`.

    Example:
        CONFIG_PATH = "/path/to/your/scenario/config.json"
    
    What happens:
    - The script processes the configuration to prepare data, preprocess sequences, 
      and save all outputs (logs, vocab files, preprocessed data, etc.) in a unique directory under the `/runs` folder.
    - Logs and preprocessed data are grouped by scenario for easier tracking of runs.
    - Look in the system_***.log to see what happened
    - If the runs folder gets bloated, feel free to delete it (given this is not vital information) the folder will be recreated when running new scenarios.

    Notes:
    - Ensure that the raw data is located in the `data/raw/` directory and matches the `fasta_file` path defined in the config.
    - To force a reprocessing of prepared data for a run, update the `force_reprocess` flag in the configuration file to `true`.

    """

    CONFIG_PATH = "/Users/filipberntsson/Documents/Studies/Thesis/Programming/BarcodeClassifier/scenarios/config_k3_base_end_end.json"  # Update with the actual path to your config file

    # Optional argumetns, for logging precision.
    system_log_level=10 #20 for info level, 10 for coarse debug, 9 for finer, 8 for superfine (warning, likely too fine)
    training_log_level=10 #no logging impleneted here as of yet

    # Run the scenario using the specified configuration
    
    
    

    run_scenario(CONFIG_PATH, system_log_level, training_log_level)


if __name__ == "__main__":
    main()

