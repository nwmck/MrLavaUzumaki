# MrLavaLoba

Stochastic model for lava flows.

## Requirements

- Python 3.x
- numpy (http://www.numpy.org/)
- scipy
- matplotlib (https://matplotlib.org/)
- pandas

Bundled dependencies (no installation required):
- `rtnorm.py` - Truncated Gaussian distribution sampling by C. Lassner (https://github.com/classner/py-rtnorm)
- `shapefile.py` - ESRI Shapefile support by jlawhead (https://github.com/GeospatialPython/pyshp)

## Usage

### Single Run

```bash
python mr_lava_loba.py
```

This runs a single simulation using parameters from `input_data.py` and `input_data_advanced.py`. See `input_data.py` for parameter definitions.

### Batch Runs

To run multiple simulations with different parameter combinations:

```bash
python run_batch.py
```

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

Results are saved to the `Run Files/` directory:
- `{run_name}_thickness_full.asc` - Complete thickness raster
- `{run_name}_thickness_masked_{threshold}.asc` - Masked thickness raster
- `{run_name}_avg_thick.txt` - Summary metrics
- `{run_name}_inp.bak` - Backup of input parameters used

