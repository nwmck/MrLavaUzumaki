# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MrLavaLoba is a stochastic lava flow simulation model. It generates probabilistic lava flow paths using a lobe-based approach where elliptical lobes propagate downslope based on terrain, inertia, and configurable probability distributions.

## Running the Model

### Single Run
```bash
python mr_lava_loba.py
```

The model reads parameters from `input_data.py` (basic parameters) and `input_data_advanced.py` (advanced parameters).

### Batch Runs
```bash
python run_batch.py
```

Processes multiple simulations defined in `Run Report.csv`. Each row with an empty "Run #" and at least one input parameter will be processed. Results are written back to the CSV.

## Dependencies

- numpy
- scipy
- matplotlib
- pandas

Bundled scripts (do not require installation):
- `rtnorm.py` - Truncated Gaussian distribution sampling
- `shapefile.py` - ESRI Shapefile read/write support

## Architecture

### Configuration System

Parameters are split into two Python files that are imported directly by `mr_lava_loba.py`:

- **`input_data.py`**: Core simulation parameters
  - DEM source file path (`source`)
  - Vent coordinates (`x_vent`, `y_vent`)
  - Volume/lobe settings (`total_volume`, `lobe_area`, `avg_lobe_thickness`)
  - Flow behavior (`n_flows`, `min_n_lobes`, `max_n_lobes`)
  - Probability controls (`lobe_exponent`, `max_slope_prob`, `inertial_exponent`)
  - `vent_flag` (0-8) controls how multiple vents/fissures are handled

- **`input_data_advanced.py`**: Secondary parameters
  - Output flags (`saveraster_flag`, `saveshape_flag`, `plot_lobes_flag`)
  - Lobe geometry (`npoints`, `aspect_ratio_coeff`, `max_aspect_ratio`)
  - Restart files for continuing from previous flows

### Main Simulation (mr_lava_loba.py)

The core algorithm:
1. Loads ASCII DEM file and parses 6-line header (cols, rows, lower-left x/y, cell size, nodata)
2. For each flow (n_flows iterations):
   - Generates lobes starting from vent location
   - Each new lobe's direction combines slope gradient and parent lobe inertia
   - Lobe shape (ellipse) varies with local slope
   - Thickness modifies the working DEM surface
3. Outputs thickness rasters (`.asc`) and optionally shapefiles

### Output Files

Simulation outputs saved to `Run Files/` directory:
- `{run_name}_thickness_full.asc` - Complete thickness raster
- `{run_name}_thickness_masked_{threshold}.asc` - Masked by volume/area threshold
- `{run_name}_avg_thick.txt` - Summary metrics

Input parameter backups saved to `Input Parameter History/` directory:
- `{run_name}_inp.bak` - Backup of input_data.py
- `{run_name}_advanced_inp.bak` - Backup of input_data_advanced.py

### Batch Runner (run_batch.py)

Orchestrates multiple simulation runs via `Run Report.csv`:
- Reads CSV and identifies rows to process (empty "Run #" + at least one input parameter)
- Temporarily modifies `input_data.py` with row-specific overrides
- Runs `mr_lava_loba.py` as subprocess
- Parses results from `*_avg_thick.txt` and writes to CSV output columns
- Fills empty input cells with defaults that were used

**CSV Structure:**
- Left columns: Input parameters (source, n_flows, avg_lobe_thickness, etc.)
- Right columns: Output metrics (Average lobe thickness, Total volume, Total area, etc.)
- "Run #" column: Empty = needs processing, Filled = completed

**Processing Logic:**
1. Load defaults from `input_data.py`
2. For each row: merge CSV overrides with defaults
3. Backup and restore `input_data.py` around each simulation
4. Parse 8 metrics from `*_avg_thick.txt` file
5. Update CSV with run name and results
6. Save after each run (progress preserved if interrupted)

### Utility Scripts

- **`make_plot.py`**: 3D visualization of terrain and flow deposits
- **`union_diff.py`**: Compare two thickness rasters (intersection/union areas, fitting parameter)

## Key Simulation Parameters

- `volume_flag`: 1 = specify total volume (calculates lobe dimensions), 0 = specify lobe dimensions directly
- `fixed_dimension_flag`: When volume_flag=1, determines if lobe area (1) or thickness (2) is fixed
- `thickening_parameter`: 0-1, controls flow channelization vs spreading
- `lobe_exponent`: 0 = new lobes from most recent, 1 = uniform probability across all lobes
- `masking_threshold`: Fraction (e.g., 0.999) for trimming low-probability output tails

## Important Implementation Notes

### Run Naming and Auto-Increment
- `mr_lava_loba.py` auto-increments run numbers by checking for existing backup files in `Input Parameter History/`
- Format: `{base_name}_{NNN}` where NNN is a 3-digit zero-padded number
- Never manually edit run names in backup files as this affects the auto-increment logic

### NumPy Compatibility
- Code has been updated to avoid NumPy 1.25+ deprecation warnings
- Use scalar returns from `np.random.uniform()` (no `size=1` parameter)
- Extract scalars from arrays with `.item()` before int() conversion

### Git Tracking
- `.gitignore` excludes `Run Files/` and `Input Parameter History/` directories
- Output files and parameter backups are never committed to git
- Only source code, documentation, and `Run Report.csv` are tracked

### Parallel Execution (TODO)
The batch runner currently runs sequentially. To parallelize:
1. Use `concurrent.futures.ProcessPoolExecutor` for multi-core execution
2. Create isolated temporary directories per worker to avoid `input_data.py` conflicts
3. Each subprocess runs `mr_lava_loba.py` independently
4. Python's GIL is not a limitation since each simulation runs as a separate subprocess
5. Alternative: modify `mr_lava_loba.py` to accept CLI arguments instead of file imports (larger refactor)

## Common Development Tasks

### Testing Changes to Simulation Code
1. Modify parameters in `input_data.py`
2. Run single simulation: `python mr_lava_loba.py`
3. Check outputs in `Run Files/` directory
4. Review parameter backup in `Input Parameter History/`

### Adding New Parameters
1. Add parameter to `input_data.py` or `input_data_advanced.py`
2. Import it in `mr_lava_loba.py` (around lines 17-44)
3. If batch-configurable, add to `INPUT_COLUMNS` list in `run_batch.py`
4. Update `Run Report.csv` header with new column
5. Update documentation in README.md

### Troubleshooting

**Batch runner can't find run results:**
- Check that backup files are being created in `Input Parameter History/`
- Verify `find_latest_run_name()` is looking in correct directory

**NumPy deprecation warnings:**
- Ensure `np.random.uniform()` calls don't use `size=1`
- Use `.item()` to extract scalars before `int()` conversion

**Run numbers not incrementing:**
- Verify backup files exist in `Input Parameter History/` folder
- Check folder creation logic in `ensure_backup_folder()`
