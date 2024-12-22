import os

def print_last_lines_of_logs(runs_dir="runs"):
    """
    Go through all system log files in the runs directory and print the last line of each log file.

    Args:
        runs_dir (str): Path to the runs directory containing scenario subfolders.
    """
    if not os.path.exists(runs_dir):
        print(f"Runs directory '{runs_dir}' does not exist.")
        return

    # Debug: Check runs directory contents
    print(f"Scanning directory: {runs_dir}")
    #print(f"Contents: {os.listdir(runs_dir)}")

    # Iterate over all subdirectories in the runs directory
    for scenario_folder in os.listdir(runs_dir):
        scenario_path = os.path.join(runs_dir, scenario_folder)

        # Debug: Check if it's a directory
        if not os.path.isdir(scenario_path):
         #   print(f"Skipping non-directory: {scenario_path}")
            continue

        #print(f"Checking scenario folder: {scenario_folder}")

        # Find the system log file in the scenario folder
        found_log = False
        for log_file in os.listdir(scenario_path):
            if log_file.startswith("system_") and log_file.endswith(".log"):
                log_file_path = os.path.join(scenario_path, log_file)
                print(f"Found log file: {log_file_path}")

                try:
                    # Read the last line of the log file
                    with open(log_file_path, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            print(f"{scenario_folder}: {last_line}")
                        else:
                            print(f"{scenario_folder}: Log file is empty.")
                    found_log = True
                except Exception as e:
                    print(f"Error reading log file '{log_file_path}': {e}")
        
        if not found_log:
            print(f"No system log file found in {scenario_folder}.")


print_last_lines_of_logs(runs_dir="runs")