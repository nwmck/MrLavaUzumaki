# MrLavaLoba

Stochastic model for lava flows.

## Requirements

- Python 3.x
- numpy (http://www.numpy.org/)
- scipy
- matplotlib (https://matplotlib.org/)
- pandas
- psutil (for RAM monitoring in batch mode)

Bundled dependencies (no installation required):
- `rtnorm.py` - Truncated Gaussian distribution sampling by C. Lassner (https://github.com/classner/py-rtnorm)
- `shapefile.py` - ESRI Shapefile support by jlawhead (https://github.com/GeospatialPython/pyshp)

## Usage

### Single Run

```bash
python mr_lava_loba.py
```

This runs a single simulation using parameters from `input_data.py` and `input_data_advanced.py`. See `input_data.py` for parameter definitions.

### Batch Runs (Parallel Execution)

To run multiple simulations with different parameter combinations:

```bash
python run_batch.py                    # Use settings from input_data.py
python run_batch.py --workers 4        # Override to use 4 workers
python run_batch.py --ignore-ram-check # Skip RAM availability check
```

#### Parallel Execution Settings

Configure in `input_data.py`:

```python
worker_mode = 'fixed'        # 'fixed' or 'auto'
n_workers = 2                # Number of workers (when mode='fixed')
cores_to_reserve = 2         # Cores to keep free (when mode='auto')
ignore_ram_check = False     # Skip RAM checks
```

- **fixed mode**: Uses exactly `n_workers` parallel processes
- **auto mode**: Uses `(total_cores - cores_to_reserve)` processes
- **RAM check**: Warns if estimated memory usage exceeds 80% of available RAM

#### How Batch Runs Work

1. **Configure runs in `Run Report.csv`**: Each row represents a potential simulation run. The CSV contains both input parameter columns and output result columns.

2. **Input columns** (left side of CSV):
   - `Run #` - Leave empty for rows you want to process
   - `source` - Path to DEM file
   - `masking_threshold`, `n_flows`, `min_n_lobes`, `volume_flag`, `total_volume`, `fixed_dimension_flag`, `lobe_area`, `avg_lobe_thickness`, `thickening_parameter`, `lobe_exponent`, `max_slope_prob`, `inertial_exponent`

3. **Output columns** (right side of CSV):
   - `Average lobe thickness`, `Total volume`, `Total area`, `Average thickness full`, `Masking threshold`, `Masked volume`, `Masked area`, `Average thickness mask`

#### Row Processing Rules

A row will be processed if:
- The `Run #` column is **empty** (rows with a Run # are considered already complete)
- At least **one input parameter** is specified

#### What Happens During Processing

For each row to process:
1. Default values are loaded from `input_data.py`
2. Any non-empty CSV values override the defaults
3. The simulation runs with the merged parameters
4. Results are written to the output columns
5. The `Run #` is populated with the generated run name (e.g., `example_run_007`)
6. Empty input cells are filled with the default values that were used

The CSV is saved after each run, so progress is preserved if the batch is interrupted.

## Output Files

Simulation outputs are saved to the **`Run Files/`** directory:
- `{run_name}_thickness_full.asc` - Complete thickness raster
- `{run_name}_thickness_masked_{threshold}.asc` - Masked thickness raster
- `{run_name}_avg_thick.txt` - Summary metrics (8 output metrics including total volume, area, average thickness)

Input parameter backups are saved to the **`Input Parameter History/`** directory:
- `{run_name}_inp.bak` - Backup of `input_data.py` parameters used for this run
- `{run_name}_advanced_inp.bak` - Backup of `input_data_advanced.py` parameters used for this run

