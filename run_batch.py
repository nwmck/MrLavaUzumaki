"""
Batch runner for MrLavaLoba simulations with parallel execution support.

Reads run configurations from 'Run Report.csv', executes simulations in parallel
using multiple CPU cores, and writes results back to the CSV.

Usage:
    python run_batch.py                    # Use settings from input_data.py
    python run_batch.py --workers 4        # Override to use 4 workers
    python run_batch.py --ignore-ram-check # Skip RAM usage checks
"""

import pandas as pd
import subprocess
import sys
import os
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import psutil
import threading
import time

# CSV file for run configurations and results
CSV_FILE = 'Run Report.csv'

# Input parameter columns (these can be overridden in CSV)
INPUT_COLUMNS = [
    'source', 'masking_threshold', 'n_flows', 'min_n_lobes', 'max_n_lobes', 'volume_flag',
    'total_volume', 'fixed_dimension_flag', 'lobe_area', 'avg_lobe_thickness',
    'thickening_parameter', 'lobe_exponent', 'max_slope_prob', 'inertial_exponent'
]

# Output columns (written after simulation completes)
OUTPUT_COLUMNS = [
    'Average lobe thickness', 'Total volume', 'Total area',
    'Average thickness full', 'Masking threshold', 'Masked volume',
    'Masked area', 'Average thickness mask'
]

# Lock for thread-safe CSV writing
csv_lock = threading.Lock()

# Global tracking for progress display
active_workers = {}
progress_lock = threading.Lock()


def load_worker_settings():
    """Load worker configuration from input_data.py"""
    settings = {
        'worker_mode': 'fixed',
        'n_workers': 2,
        'cores_to_reserve': 2,
        'ignore_ram_check': False
    }

    try:
        with open('input_data.py', 'r') as f:
            content = f.read()

        for key in settings.keys():
            pattern = rf'^{key}\s*=\s*(.+?)(?:\s*#.*)?$'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                value_str = match.group(1).strip().split('#')[0].strip()
                try:
                    settings[key] = eval(value_str)
                except:
                    settings[key] = value_str.strip("'\"")
    except FileNotFoundError:
        print("Warning: input_data.py not found, using default worker settings")

    return settings


def calculate_workers(settings, cli_override=None):
    """Calculate the number of workers to use."""
    total_cores = multiprocessing.cpu_count()

    if cli_override is not None:
        if cli_override == -1:
            return total_cores
        return cli_override

    if settings['worker_mode'] == 'auto':
        workers = max(1, total_cores - settings['cores_to_reserve'])
    else:
        workers = settings['n_workers']

    return min(workers, total_cores)


def estimate_memory_per_simulation(dem_path):
    """Estimate memory usage per simulation based on DEM file size."""
    try:
        if os.path.exists(dem_path):
            file_size = os.path.getsize(dem_path)
            # Estimate: DEM loaded + output arrays + overhead
            # Rough multiplier: 10x file size for in-memory arrays
            estimated_mb = (file_size * 10) / (1024 * 1024)
            return max(100, estimated_mb)  # Minimum 100MB estimate
    except:
        pass
    return 500  # Default estimate: 500MB per simulation


def check_ram_availability(n_workers, dem_path, ignore_check=False):
    """
    Check if there's enough RAM for the requested number of workers.
    Returns (is_ok, message, recommended_workers)
    """
    if ignore_check:
        return True, "RAM check skipped (ignore_ram_check=True)", n_workers

    try:
        available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
        total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
        mem_per_sim = estimate_memory_per_simulation(dem_path)

        # Use 80% of available RAM as safety margin
        usable_ram = available_ram_mb * 0.8
        recommended_workers = max(1, int(usable_ram / mem_per_sim))
        required_ram = n_workers * mem_per_sim

        if required_ram <= usable_ram:
            return True, f"RAM check passed ({required_ram:.0f}MB needed, {usable_ram:.0f}MB available)", n_workers

        message = (
            f"\n{'='*60}\n"
            f"RAM WARNING\n"
            f"{'='*60}\n"
            f"Requested workers: {n_workers}\n"
            f"Estimated memory per simulation: {mem_per_sim:.0f} MB\n"
            f"Total RAM required: {required_ram:.0f} MB\n"
            f"Available RAM (80% safety): {usable_ram:.0f} MB\n"
            f"Total system RAM: {total_ram_mb:.0f} MB\n"
            f"\nRecommended max workers: {recommended_workers}\n"
            f"{'='*60}"
        )
        return False, message, recommended_workers

    except Exception as e:
        return True, f"Could not check RAM: {e}", n_workers


def load_defaults_from_input_data():
    """Load default parameter values from input_data.py"""
    defaults = {}

    with open('input_data.py', 'r') as f:
        content = f.read()

    for col in INPUT_COLUMNS:
        pattern = rf'^{col}\s*=\s*(.+?)(?:\s*#.*)?$'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            value_str = match.group(1).strip().split('#')[0].strip()
            try:
                defaults[col] = eval(value_str)
            except:
                defaults[col] = value_str

    return defaults


def parse_avg_thick_file(filepath):
    """Parse the avg_thick.txt file and extract metrics"""
    results = {}

    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r') as f:
        content = f.read()

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


def build_cli_args(params, run_name):
    """Build command line arguments for mr_lava_loba.py"""
    args = [sys.executable, 'mr_lava_loba.py']

    # Add run name
    args.extend(['--run-name', run_name])

    # Map CSV column names to CLI argument names
    param_to_cli = {
        'source': '--source',
        'masking_threshold': '--masking-threshold',
        'n_flows': '--n-flows',
        'min_n_lobes': '--min-n-lobes',
        'max_n_lobes': '--max-n-lobes',
        'volume_flag': '--volume-flag',
        'total_volume': '--total-volume',
        'fixed_dimension_flag': '--fixed-dimension-flag',
        'lobe_area': '--lobe-area',
        'avg_lobe_thickness': '--avg-lobe-thickness',
        'thickening_parameter': '--thickening-parameter',
        'lobe_exponent': '--lobe-exponent',
        'max_slope_prob': '--max-slope-prob',
        'inertial_exponent': '--inertial-exponent',
    }

    for param, cli_arg in param_to_cli.items():
        if param in params and params[param] is not None:
            # Special handling for max_n_lobes: skip if it's the string "min_n_lobes"
            # This allows the auto-logic in mr_lava_loba.py to set max = min
            if param == 'max_n_lobes' and str(params[param]).strip().lower() == 'min_n_lobes':
                continue
            args.extend([cli_arg, str(params[param])])

    return args


def generate_run_name(base_name='batch_run'):
    """Generate a unique run name based on existing backup files."""
    import glob

    i = 0
    while True:
        run_name = f"{base_name}_{i:03d}"
        backup_file = os.path.join('Input Parameter History', f'{run_name}_inp.bak')
        if not os.path.exists(backup_file):
            return run_name
        i += 1


def run_single_simulation(task):
    """
    Run a single simulation. This function is called by worker processes.
    Returns (row_index, results_dict)
    """
    row_index, row_params, defaults, run_name = task

    # Merge defaults with row-specific overrides
    params = defaults.copy()
    for col in INPUT_COLUMNS:
        if col in row_params and pd.notna(row_params[col]) and str(row_params[col]).strip() != '':
            params[col] = row_params[col]

    # Replace "min_n_lobes" placeholder in max_n_lobes with actual numeric value
    if 'max_n_lobes' in params and str(params['max_n_lobes']).strip().lower() == 'min_n_lobes':
        if 'min_n_lobes' in params:
            params['max_n_lobes'] = params['min_n_lobes']

    # Build CLI arguments
    args = build_cli_args(params, run_name)

    try:
        # Run the simulation
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        # Parse the results from the avg_thick.txt file
        avg_thick_path = os.path.join('Run Files', f'{run_name}_avg_thick.txt')

        # Wait briefly for file to be written
        for _ in range(10):
            if os.path.exists(avg_thick_path):
                break
            time.sleep(0.5)

        results = parse_avg_thick_file(avg_thick_path)
        results['run_name'] = run_name
        results['params_used'] = params
        results['success'] = True
        results['stdout'] = result.stdout
        results['stderr'] = result.stderr

    except Exception as e:
        results = {
            'run_name': run_name,
            'params_used': params,
            'success': False,
            'error': str(e)
        }

    return row_index, results


def should_process_row(row):
    """Determine if a row should be processed."""
    has_run_number = pd.notna(row.get('Run #')) and str(row.get('Run #')).strip() != ''

    if has_run_number:
        return False

    for col in INPUT_COLUMNS:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
            return True

    return False


def update_progress_display(active, total_rows, completed):
    """Update the progress display showing active workers."""
    with progress_lock:
        status_lines = []
        for worker_id, (row_idx, run_name) in active.items():
            status_lines.append(f"  Worker {worker_id}: Row {row_idx + 1} ({run_name})")

        if status_lines:
            print(f"\rProgress: {completed}/{total_rows} complete | Active workers:", end='')
            print('\n' + '\n'.join(status_lines), end='\r')


def main():
    """Main batch runner function with parallel execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Batch runner for MrLavaLoba')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (overrides input_data.py)')
    parser.add_argument('--ignore-ram-check', action='store_true',
                        help='Skip RAM availability check')
    args = parser.parse_args()

    # Check if CSV exists
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        sys.exit(1)

    # Load settings
    worker_settings = load_worker_settings()
    ignore_ram = args.ignore_ram_check or worker_settings.get('ignore_ram_check', False)

    # Calculate number of workers
    n_workers = calculate_workers(worker_settings, args.workers)
    total_cores = multiprocessing.cpu_count()

    print(f"\n{'='*60}")
    print("MrLavaLoba Batch Runner - Parallel Execution")
    print(f"{'='*60}")
    print(f"Total CPU cores: {total_cores}")
    print(f"Worker mode: {worker_settings['worker_mode']}")
    print(f"Workers to use: {n_workers}")

    # Load the CSV
    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')

    # Load defaults
    defaults = load_defaults_from_input_data()
    print(f"\nLoaded {len(defaults)} default parameters from input_data.py")

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

    total_rows = len(rows_to_process)
    print(f"\nFound {total_rows} row(s) to process")

    # Check RAM availability
    dem_path = defaults.get('source', '')
    ram_ok, ram_message, recommended = check_ram_availability(n_workers, dem_path, ignore_ram)
    print(f"\n{ram_message}")

    if not ram_ok:
        print(f"\nOptions:")
        print(f"  1. Continue with {n_workers} workers (may cause system slowdown)")
        print(f"  2. Use recommended {recommended} workers")
        print(f"  3. Abort")

        try:
            choice = input("\nEnter choice (1/2/3): ").strip()
            if choice == '2':
                n_workers = recommended
                print(f"Using {n_workers} workers")
            elif choice == '3':
                print("Aborted.")
                return
            else:
                print(f"Continuing with {n_workers} workers")
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    # Prepare tasks - generate unique run names SEQUENTIALLY to avoid race conditions
    tasks = []
    # Find the starting number by checking existing backup files
    import glob
    existing_batch_runs = glob.glob(os.path.join('Input Parameter History', 'batch_run_*_inp.bak'))
    if existing_batch_runs:
        # Extract numbers from filenames like "batch_run_042_inp.bak"
        numbers = []
        for f in existing_batch_runs:
            match = re.search(r'batch_run_(\d+)_inp\.bak', os.path.basename(f))
            if match:
                numbers.append(int(match.group(1)))
        start_num = max(numbers) + 1 if numbers else 0
    else:
        start_num = 0

    # Generate unique names sequentially
    for i, idx in enumerate(rows_to_process):
        row = df.iloc[idx]
        run_name = f'batch_run_{start_num + i:03d}'
        tasks.append((idx, row.to_dict(), defaults, run_name))

    print(f"\n{'='*60}")
    print(f"Starting parallel execution with {n_workers} workers")
    print(f"{'='*60}\n")

    completed = 0
    start_time = time.time()

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_single_simulation, task): task for task in tasks}

        # Process results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            row_index = task[0]
            run_name = task[3]

            try:
                idx, results = future.result()
                completed += 1

                if results.get('success', False) and results.get('run_name'):
                    # Update the DataFrame
                    with csv_lock:
                        df.at[idx, 'Run #'] = results['run_name']

                        for col in OUTPUT_COLUMNS:
                            if col in results:
                                df.at[idx, col] = results[col]

                        params_used = results.get('params_used', {})
                        for col in INPUT_COLUMNS:
                            if col in df.columns:
                                current_val = df.at[idx, col]
                                # Fill in empty values or replace placeholder text
                                should_update = (
                                    pd.isna(current_val) or
                                    str(current_val).strip() == '' or
                                    (col == 'max_n_lobes' and str(current_val).strip().lower() == 'min_n_lobes')
                                )
                                if should_update and col in params_used:
                                    df.at[idx, col] = params_used[col]

                        # Save CSV after each completion
                        df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')

                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_rows - completed) / rate if rate > 0 else 0

                    print(f"[{completed}/{total_rows}] Row {idx + 1} completed: {results['run_name']} "
                          f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
                else:
                    print(f"\n{'='*60}")
                    print(f"[{completed}/{total_rows}] Row {row_index + 1} FAILED: {results.get('run_name', 'unknown')}")
                    print(f"Error: {results.get('error', 'Unknown error')}")
                    if 'stderr' in results and results['stderr']:
                        print(f"STDERR:\n{results['stderr']}")
                    if 'stdout' in results and results['stdout']:
                        print(f"STDOUT:\n{results['stdout']}")
                    print(f"{'='*60}\n")

            except Exception as e:
                completed += 1
                print(f"[{completed}/{total_rows}] Row {row_index + 1} EXCEPTION: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Processed: {completed} simulations")
    print(f"Average: {elapsed/completed:.1f} seconds per simulation")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
