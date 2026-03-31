# track_map

Interactive viewer for tropical cyclone tracks and landfalls generated
by [ExtremeWeatherBench](https://github.com/brightbandtech/ExtremeWeatherBench).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd ~/code/ewb_tc_track_mapper
uv sync
```

The project depends on the local `ExtremeWeatherBench` checkout at
`~/code/ExtremeWeatherBench` (editable install, `fix/tc-tracker` branch).

## Usage

### Start the dashboard

```bash
.venv/bin/panel serve app.py --port 9999 --allow-websocket-origin='*'
```

Then open `http://localhost:9999/app` in your browser (use an SSH tunnel
if running on a remote VM: `ssh -L 9999:localhost:9999 <host>`).

### Pre-generate a NetCDF file

The dashboard can generate on the fly, but for large models it's faster
to pre-generate:

```bash
.venv/bin/python generate.py --case-id 163 --model GRAP_v100_GFS
```

Output is saved to `data/<model>/case_<id>.nc`.

List available TC cases:

```bash
.venv/bin/python generate.py --list-cases
```

### Supported models

All CIRA icechunk models plus `HRES`. Run `--list-cases` to browse TC
case IDs.

## Dashboard controls

| Control | Description |
|---|---|
| **Case** | EWB tropical cyclone case |
| **Model** | Forecast model |
| **View** | Load an already-generated NetCDF and display it |
| **Generate & View** | Run the EWB pipeline, save NetCDF, then display |

The map shows the IBTrACS observed track (black), one coloured forecast
track per initialisation time, and landfall markers (filled circles).

## Output format

Each NetCDF contains:

| Variable | Dims | Description |
|---|---|---|
| `detection_lat/lon` | `detection` | Forecast TC centre positions |
| `detection_slp/wind` | `detection` | SLP and wind at each detection |
| `detection_valid_time` | `detection` | Valid time of each detection |
| `detection_init_time` | `detection` | Init time of each detection |
| `observed_lat/lon` | `obs_time` | IBTrACS track positions |
| `observed_slp/wind` | `obs_time` | SLP and wind along observed track |
| `fc_landfall_*` | `fc_landfall` | Forecast landfall points |
| `obs_landfall_*` | `obs_landfall` | Observed landfall points |
