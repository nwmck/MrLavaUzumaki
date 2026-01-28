"""
Batch runner for MrLavaLoba simulations.

Reads run configurations from 'Run Report.csv', executes simulations for rows
that need processing, and writes results back to the CSV.

Usage:
    python run_batch.py

TODO: Parallel Execution
    Python's GIL (Global Interpreter Lock) only limits multi-threading for CPU-bound
    tasks, but since we run each simulation as a separate subprocess, we can easily
    parallelize using multiprocessing. Options for implementation:

    1. concurrent.futures.ProcessPoolExecutor - Simplest approach, run N simulations
       at once where N = number of CPU cores. Each worker would need its own copy of
       input_data.py to avoid conflicts.

    2. multiprocessing.Pool - Similar to above, more control over worker processes.

    3. Scheduler approach - A manager process that monitors running simulations and
       launches new ones as slots become available. Good for long-running batches.

    Key consideration: Each parallel run needs isolated input_data.py files to avoid
    race conditions. Could use:
    - Temporary directories per worker
    - Pass parameters via command-line args (requires modifying mr_lava_loba.py)
    - Environment variables
"""

import pandas as pd
import subprocess
import sys
import os
import re
import shutil
from pathlib import Path

# CSV file for run configurations and results
CSV_FILE = 'Run Report.csv'

# Input parameter columns (these can be overridden in CSV)
INPUT_COLUMNS = [
    'source', 'masking_threshold', 'n_flows', 'min_n_lobes', 'volume_flag',
    'total_volume', 'fixed_dimension_flag', 'lobe_area', 'avg_lobe_thickness',
    'thickening_parameter', 'lobe_exponent', 'max_slope_prob', 'inertial_exponent'
]

# Output columns (written after simulation completes)
OUTPUT_COLUMNS = [
    'Average lobe thickness', 'Total volume', 'Total area',
    'Average thickness full', 'Masking threshold', 'Masked volume',
    'Masked area', 'Average thickness mask'
]


def load_defaults_from_input_data():
    """Load default parameter values from input_data.py"""
    defaults = {}

    # Read input_data.py and extract variable assignments
    with open('input_data.py', 'r') as f:
        content = f.read()

    # Parse each input column from the file
    for col in INPUT_COLUMNS:
        # Match patterns like: variable_name = value
        pattern = rf'^{col}\s*=\s*(.+?)(?:\s*#.*)?$'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            value_str = match.group(1).strip()
            # Try to evaluate the value (handles numbers, strings, etc.)
            try:
                # Remove trailing comments
                value_str = value_str.split('#')[0].strip()
                defaults[col] = eval(value_str)
            except:
                defaults[col] = value_str

    return defaults


def create_temp_input_file(params, original_file='input_data.py', temp_file='input_data_temp.py'):
    """Create a temporary input_data.py with overridden parameters"""
    with open(original_file, 'r') as f:
        content = f.read()

    # Replace each parameter value in the file
    for key, value in params.items():
        if key in INPUT_COLUMNS:
            # Format the value appropriately
            if isinstance(value, str):
                value_str = f'"{value}"'
            else:
                value_str = str(value)

            # Replace the line with the parameter
            pattern = rf'^({key}\s*=\s*)(.+?)(\s*#.*)?$'
            replacement = rf'\g<1>{value_str}\g<3>' if re.search(pattern, content, re.MULTILINE) else None

            if replacement:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    with open(temp_file, 'w') as f:
        f.write(content)

    return temp_file


def parse_avg_thick_file(filepath):
    """Parse the avg_thick.txt file and extract metrics"""
    results = {}

    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse each metric
    patterns = {
        'Average lobe thickness': r'Average lobe thickness = ([\d.e+-]+)',
        'Total volume': r'Total volume = ([\d.e+-]+)',
        'Total area': r'Total area = ([\d.e+-]+)',
        'Average thickness full': r'Average thickness full = ([\d.e+-]+)',
        'Masking threshold': r'Masking threshold = ([\d.e+-]+)',
        'Masked volume': r'Masked volume = ([\d.e+-]+)',
        'Masked area': r'Masked area = ([\d.e+-]+)',
        'Average thickness mask': r'Average thickness mask = ([\d.e+-]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))

    return results


def find_latest_run_name():
    """Find the most recently created run name from backup files"""
    import glob

    # Look for the newest backup file in the Input Parameter History folder
    backup_files = glob.glob(os.path.join('Input Parameter History', '*_inp.bak'))
    if backup_files:
        # Sort by modification time, get newest
        newest = max(backup_files, key=os.path.getmtime)
        # Extract run name (remove path and _inp.bak suffix)
        filename = os.path.basename(newest)
        return filename.replace('_inp.bak', '')

    return None


def run_simulation(row_params, row_index, defaults):
    """Run a single simulation with the given parameters"""

    # Merge defaults with row-specific overrides
    params = defaults.copy()
    for col in INPUT_COLUMNS:
        if col in row_params and pd.notna(row_params[col]) and row_params[col] != '':
            params[col] = row_params[col]

    # Backup original input_data.py
    shutil.copy('input_data.py', 'input_data.py.backup')

    try:
        # Modify input_data.py with our parameters
        with open('input_data.py', 'r') as f:
            original_content = f.read()

        modified_content = original_content
        for key, value in params.items():
            if key in INPUT_COLUMNS:
                # Format the value appropriately
                if isinstance(value, str):
                    value_str = f'"{value}"'
                else:
                    value_str = str(value)

                # Replace the line with the parameter
                pattern = rf'^({key}\s*=\s*)(.+?)(\s*#.*)?$'
                if re.search(pattern, modified_content, re.MULTILINE):
                    modified_content = re.sub(
                        pattern,
                        rf'\g<1>{value_str}  \g<3>',
                        modified_content,
                        flags=re.MULTILINE
                    )

        with open('input_data.py', 'w') as f:
            f.write(modified_content)

        # Run the simulation
        print(f"\n{'='*60}")
        print(f"Running simulation for row {row_index + 1}")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, 'mr_lava_loba.py'],
            capture_output=False,
            text=True
        )

        # Find the run name that was just created
        run_name = find_latest_run_name()

        # Parse the results from the avg_thick.txt file
        if run_name:
            avg_thick_path = os.path.join('Run Files', f'{run_name}_avg_thick.txt')
            results = parse_avg_thick_file(avg_thick_path)
            results['run_name'] = run_name
            results['params_used'] = params
        else:
            results = {'run_name': None, 'params_used': params}

        return results

    finally:
        # Restore original input_data.py
        shutil.copy('input_data.py.backup', 'input_data.py')
        os.remove('input_data.py.backup')


def should_process_row(row):
    """
    Determine if a row should be processed.
    Criteria: No Run # AND at least one input variable specified.
    """
    # Check if Run # is empty
    has_run_number = pd.notna(row.get('Run #')) and str(row.get('Run #')).strip() != ''

    if has_run_number:
        return False

    # Check if at least one input variable is specified
    for col in INPUT_COLUMNS:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
            return True

    return False


def main():
    """Main batch runner function"""

    # Check if CSV exists
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        print("Please create the CSV file with appropriate headers.")
        sys.exit(1)

    # Load the CSV (encoding handles BOM character from Excel)
    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')

    # Load defaults from input_data.py
    defaults = load_defaults_from_input_data()
    print("Loaded defaults from input_data.py:")
    for key, value in defaults.items():
        print(f"  {key}: {value}")

    # Find rows to process
    rows_to_process = []
    for idx, row in df.iterrows():
        if should_process_row(row):
            rows_to_process.append(idx)

    if not rows_to_process:
        print("\nNo rows to process. Rows need:")
        print("  - Empty 'Run #' column")
        print("  - At least one input parameter specified")
        return

    print(f"\nFound {len(rows_to_process)} row(s) to process: {[i+1 for i in rows_to_process]}")

    # Process each row
    for idx in rows_to_process:
        row = df.iloc[idx]

        # Run the simulation
        results = run_simulation(row.to_dict(), idx, defaults)

        if results.get('run_name'):
            # Update the Run # column
            df.at[idx, 'Run #'] = results['run_name']

            # Update output columns with results
            for col in OUTPUT_COLUMNS:
                if col in results:
                    df.at[idx, col] = results[col]

            # Fill empty input cells with the values that were used
            params_used = results.get('params_used', {})
            for col in INPUT_COLUMNS:
                if col in df.columns:
                    current_val = df.at[idx, col]
                    if pd.isna(current_val) or str(current_val).strip() == '':
                        if col in params_used:
                            df.at[idx, col] = params_used[col]

            print(f"\nRow {idx + 1} completed: {results['run_name']}")
        else:
            print(f"\nWarning: Row {idx + 1} - Could not determine run name")

        # Save CSV after each run (in case of interruption)
        df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"Updated {CSV_FILE}")

    print(f"\n{'='*60}")
    print("Batch processing complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
