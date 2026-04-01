# TC Track Mapper for ExtremeWeatherBench

Interactive viewer for tropical cyclone tracks and landfalls generated
by [ExtremeWeatherBench](https://github.com/brightbandtech/ExtremeWeatherBench).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- A local clone of ExtremeWeatherBench on the `develop` branch (soon will be >=1.0.4)

## Setup

### 1. Clone ExtremeWeatherBench (soon will be fine using PyPI)

```bash
git clone https://github.com/brightbandtech/ExtremeWeatherBench.git ~/code/ExtremeWeatherBench
cd ~/code/ExtremeWeatherBench
git checkout develop
```

### 2. Clone this repo and create the virtual environment using uv

```bash
git clone https://github.com/aaTman/ewb_tc_track_mapper.git ~/code/ewb_tc_track_mapper
cd ~/code/ewb_tc_track_mapper
uv sync
```

This creates `.venv/` and installs all dependencies, including
`ExtremeWeatherBench` as an editable local install from `~/code/ExtremeWeatherBench`.

> **Note:** if your EWB checkout is in a different location, update the
> `[tool.uv.sources]` entry in `pyproject.toml` before running `uv sync`.

## Running the dashboard

### Local machine

```bash
cd ~/code/ewb_tc_track_mapper
.venv/bin/panel serve app.py --port 9999 --allow-websocket-origin='*'
```

Open `http://localhost:9999/app` in your browser.

### Remote VM (SSH tunnel)

On your local machine, open the tunnel:

```bash
ssh -L 9999:localhost:9999 <user>@<vm-host>
```

Then on the VM start the server as above and open `http://localhost:9999/app`
locally.

## Generating track data

The dashboard can generate data on the fly via the **Generate & View** button,
but for large models or bulk pre-generation use the CLI scripts below.

### Single case

```bash
.venv/bin/python generate.py --case-id 163 --model GRAP_v100_GFS
```

Output is saved to `data/<model>/case_<id>.nc`.

### List available TC cases

```bash
.venv/bin/python generate.py --list-cases
```

### Batch generation (all cases × all models)

```bash
# default: 8 parallel workers, skip already-generated files
.venv/bin/python generate_all.py --skip-existing

# custom worker count
.venv/bin/python generate_all.py --workers 4 --skip-existing

# specific models only
.venv/bin/python generate_all.py --models GRAP_v100_GFS FOUR_v200_GFS

# specific cases only
.venv/bin/python generate_all.py --case-ids 161 163 201
```

Transient network errors are retried automatically (3 attempts by default).
Cases where a model has no data coverage are reported in the summary but
are expected — not all models ran for every storm.

## Dashboard controls

| Control | Description |
|---|---|
| **Case** | EWB tropical cyclone case |
| **Model** | Forecast model |
| **✅ / ⚫ indicator** | Shows whether a NetCDF already exists for the selection |
| **View** | Load an already-generated NetCDF and display it |
| **Generate & View** | Run the EWB pipeline, save NetCDF, then display |

The map shows:
- **Black line** — IBTrACS observed track
- **Coloured lines** — one forecast track per initialisation time (rainbow,
  violet = earliest → red = latest)
- **Black circles** — observed landfall points
- **Red circles** — forecast landfall points
- **Legend** — bottom-right corner of the map

## Output format

Each NetCDF contains:

| Variable | Dims | Description |
|---|---|---|
| `detection_lat/lon` | `detection` | Forecast TC center positions |
| `detection_slp/wind` | `detection` | SLP and wind at each detection |
| `detection_valid_time` | `detection` | Valid time of each detection |
| `detection_init_time` | `detection` | Init time of each detection |
| `observed_lat/lon` | `obs_time` | IBTrACS track positions |
| `observed_slp/wind` | `obs_time` | SLP and wind along observed track |
| `fc_landfall_*` | `fc_landfall` | Forecast landfall points |
| `obs_landfall_*` | `obs_landfall` | Observed landfall points |

## Supported models

All CIRA icechunk model names (see `extremeweatherbench.inputs.CIRA_MODEL_NAMES`)
plus `HRES` (from WeatherBench2 on GCS, public anonymous access). Soon to be more!