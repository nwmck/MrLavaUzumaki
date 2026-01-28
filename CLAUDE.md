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

### Utility Scripts

- **`make_plot.py`**: 3D visualization of terrain and flow deposits
- **`union_diff.py`**: Compare two thickness rasters (intersection/union areas, fitting parameter)

## Key Simulation Parameters

- `volume_flag`: 1 = specify total volume (calculates lobe dimensions), 0 = specify lobe dimensions directly
- `fixed_dimension_flag`: When volume_flag=1, determines if lobe area (1) or thickness (2) is fixed
- `thickening_parameter`: 0-1, controls flow channelization vs spreading
- `lobe_exponent`: 0 = new lobes from most recent, 1 = uniform probability across all lobes
- `masking_threshold`: Fraction (e.g., 0.999) for trimming low-probability output tails
