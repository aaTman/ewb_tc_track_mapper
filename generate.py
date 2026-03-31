"""Generate TC track netcdf files for the interactive viewer.

Usage:
    python generate.py --case-id 163 --model GRAP_v100_GFS
    python generate.py --case-id 152 --model FOUR_v200_GFS --output-dir ./data

The output file is saved to:
    <output_dir>/<model>/case_<case_id:03d>.nc

The netcdf stores forecast track detections (one row per detected TC
center), the observed IBTrACS track, and landfall points for both.
"""

import argparse
import logging
import pathlib

import numpy as np
import xarray as xr

import extremeweatherbench as ewb
from extremeweatherbench import calc, derived, inputs
from extremeweatherbench.inputs import CIRA_MODEL_NAMES

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = pathlib.Path(__file__).parent / "data"
ALL_MODELS = list(CIRA_MODEL_NAMES) + ["HRES"]

HRES_SOURCE = (
    "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
)


def _run_pipeline(case_metadata, input_data, **kwargs):
    """Drive the EWB pipeline without the check_for_missing_data gate.

    EWB's run_pipeline calls check_for_valid_times, which relies on
    xarray raising KeyError when .sel() is called on a non-indexed
    coordinate. Newer xarray (>=2026) returns an empty array instead,
    causing check_for_valid_times to return False for CIRA icechunk
    data even when init_times cover the case range.

    This helper replicates every step of run_pipeline except that gate.
    """
    import extremeweatherbench.sources as sources

    data = input_data.open_and_maybe_preprocess_data_from_source().pipe(
        lambda ds: input_data.maybe_map_variable_names(ds)
    )
    source_module = sources.get_backend_module(type(data))
    return (
        inputs.maybe_subset_variables(
            data,
            variables=input_data.variables,
            source_module=source_module,
        )
        .pipe(lambda ds: input_data.subset_data_to_case(ds, case_metadata, **kwargs))
        .pipe(input_data.maybe_convert_to_dataset)
        .pipe(input_data.add_source_to_dataset_attrs)
        .pipe(
            lambda ds: derived.maybe_derive_variables(
                ds,
                variables=input_data.variables,
                case_metadata=case_metadata,
                **kwargs,
            )
        )
    )


# ── forecast factory ──────────────────────────────────────────────────────────

def build_forecast(model_name: str):
    """Build the appropriate EWB forecast input object for *model_name*."""
    tc_vars = ewb.TropicalCycloneTrackVariables(
        output_variables=["surface_wind_speed", "air_pressure_at_mean_sea_level"]
    )
    if model_name == "HRES":
        return ewb.ZarrForecast(
            name="HRES",
            source=HRES_SOURCE,
            variables=[tc_vars],
            variable_mapping=ewb.HRES_metadata_variable_mapping,
            storage_options={"remote_options": {"anon": True}},
            preprocess=ewb.defaults.preprocess_hres_tc_forecast_dataset,
        )
    if model_name not in CIRA_MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {ALL_MODELS}"
        )
    return ewb.inputs.get_cira_icechunk(
        model_name=model_name,
        variables=[tc_vars],
        name=model_name,
        preprocess=ewb.defaults.preprocess_cira_icechunk_tc_forecast_dataset,
    )


# ── netcdf helpers ────────────────────────────────────────────────────────────

def _extract_forecast_detections(forecast_ds: xr.Dataset) -> xr.Dataset:
    """Flatten 2-D tctracks grid to a 1-D detection dataset.

    forecast_ds is the output of _run_pipeline with
    TropicalCycloneTrackVariables. It has dims (lead_time, valid_time) and
    2-D coordinates latitude/longitude that are NaN for non-TC grid points.

    Returns an xr.Dataset with dim 'detection', one row per non-NaN TC
    center. Includes detection_init_time computed as valid_time - lead_time.
    """
    slp = forecast_ds["air_pressure_at_mean_sea_level"]
    wind = forecast_ds["surface_wind_speed"]
    lat = forecast_ds["latitude"]
    lon = forecast_ds["longitude"]

    valid = ~np.isnan(slp.values) & ~np.isnan(lat.values)
    lt_idx, vt_idx = np.where(valid)

    lead_times = slp.lead_time.values[lt_idx]
    valid_times = slp.valid_time.values[vt_idx]

    return xr.Dataset(
        {
            "detection_lat": ("detection", lat.values[lt_idx, vt_idx]),
            "detection_lon": ("detection", lon.values[lt_idx, vt_idx]),
            "detection_slp": ("detection", slp.values[lt_idx, vt_idx]),
            "detection_wind": ("detection", wind.values[lt_idx, vt_idx]),
            "detection_lead_time": ("detection", lead_times),
            "detection_valid_time": ("detection", valid_times),
            "detection_init_time": ("detection", valid_times - lead_times),
        }
    )


def _extract_observed_track(target_ds: xr.Dataset) -> xr.Dataset:
    """Package the IBTrACS target dataset into an obs-prefixed Dataset."""
    slp = target_ds["air_pressure_at_mean_sea_level"]
    wind = target_ds["surface_wind_speed"]
    lat = target_ds["latitude"]
    lon = target_ds["longitude"]

    return xr.Dataset(
        {
            "observed_lat": ("obs_time", lat.values),
            "observed_lon": ("obs_time", lon.values),
            "observed_slp": ("obs_time", slp.values),
            "observed_wind": ("obs_time", wind.values),
            "observed_valid_time": ("obs_time", target_ds.valid_time.values),
        }
    )


def _landfall_to_arrays(
    lf_da: xr.DataArray,
    prefix: str,
) -> dict:
    """Extract lat/lon/time/slp arrays from a find_landfalls DataArray.

    Returns a dict mapping variable name -> (dims, values) suitable for
    passing to xr.Dataset().  Returns an empty dict when no landfalls exist.

    For forecast data find_landfalls returns a 2D (init_time × landfall)
    array that is mostly NaN (one non-NaN per init_time along the
    diagonal).  We flatten it by keeping only the non-NaN entries so
    the resulting variables are 1D and NetCDF-compatible.
    """
    if lf_da is None or len(lf_da) == 0:
        return {}

    dim = f"{prefix}_landfall"
    slp_vals = lf_da.values
    lats = lf_da.coords["latitude"].values
    lons = lf_da.coords["longitude"].values
    times = lf_da.coords["valid_time"].values

    # For 2D (init_time × landfall) forecast output, extract non-NaN entries.
    # Coords (latitude, longitude, valid_time) remain 1D indexed by landfall,
    # so use column indices, not a 2D boolean mask, to index them.
    if slp_vals.ndim == 2:
        row_idx, col_idx = np.where(~np.isnan(slp_vals))
        slp_vals = slp_vals[row_idx, col_idx]
        lats = lats[col_idx]
        lons = lons[col_idx]
        times = times[col_idx]

    if len(slp_vals) == 0:
        return {}

    return {
        f"{prefix}_landfall_lat": (dim, lats),
        f"{prefix}_landfall_lon": (dim, lons),
        f"{prefix}_landfall_time": (dim, times),
        f"{prefix}_landfall_slp": (dim, slp_vals),
    }


# ── main generation routine ───────────────────────────────────────────────────

def generate(
    case_id: int,
    model_name: str,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
) -> pathlib.Path:
    """Run the EWB pipeline and save TC tracks + landfalls to netcdf.

    Args:
        case_id: EWB case_id_number (tropical_cyclone event).
        model_name: One of CIRA_MODEL_NAMES or "HRES".
        output_dir: Root directory for output netcdf files.

    Returns:
        Path to the saved netcdf file.
    """
    # ── 1. find case metadata ─────────────────────────────────────────────
    all_cases = ewb.load_cases()
    tc_cases = [c for c in all_cases if c.event_type == "tropical_cyclone"]
    case_metadata = next(
        (c for c in tc_cases if c.case_id_number == case_id), None
    )
    if case_metadata is None:
        available = sorted(c.case_id_number for c in tc_cases)
        raise ValueError(
            f"Case {case_id} not found among TC cases. "
            f"Available IDs: {available}"
        )
    logger.info(
        "Generating tracks for case %d (%s) with model %s",
        case_id,
        case_metadata.title,
        model_name,
    )

    # ── 2. run target (IBTrACS) pipeline ──────────────────────────────────
    target = ewb.IBTrACS()
    logger.info("Running IBTrACS pipeline...")
    target_ds = _run_pipeline(case_metadata, target)

    if not target_ds.data_vars:
        raise RuntimeError(
            f"IBTrACS returned empty dataset for case {case_id}. "
            "Check that the case title matches IBTrACS storm names."
        )

    # ── 3. run forecast pipeline (passes target for TC filtering) ─────────
    forecast = build_forecast(model_name)
    logger.info("Running %s forecast pipeline...", model_name)
    forecast_ds = _run_pipeline(
        case_metadata, forecast, _target_dataset=target_ds
    )

    if not forecast_ds.data_vars:
        raise RuntimeError(
            f"Forecast returned empty dataset for case {case_id} / "
            f"model {model_name}. "
            "Check that the model covers the case time range."
        )

    # ── 4. find landfalls ─────────────────────────────────────────────────
    fc_slp_da = forecast_ds["air_pressure_at_mean_sea_level"]
    obs_slp_da = target_ds["air_pressure_at_mean_sea_level"]

    logger.info("Finding forecast landfalls...")
    try:
        fc_landfalls = calc.find_landfalls(fc_slp_da)
    except Exception as exc:
        logger.warning("Forecast landfall detection failed: %s", exc)
        fc_landfalls = None

    logger.info("Finding observed landfalls...")
    try:
        obs_landfalls = calc.find_landfalls(obs_slp_da)
    except Exception as exc:
        logger.warning("Observed landfall detection failed: %s", exc)
        obs_landfalls = None

    # ── 5. build combined netcdf dataset ──────────────────────────────────
    det_ds = _extract_forecast_detections(forecast_ds)
    obs_ds = _extract_observed_track(target_ds)

    lf_vars: dict = {}
    lf_vars.update(_landfall_to_arrays(fc_landfalls, prefix="fc"))
    lf_vars.update(_landfall_to_arrays(obs_landfalls, prefix="obs"))

    combined = xr.merge([det_ds, obs_ds])
    if lf_vars:
        combined = xr.merge([combined, xr.Dataset(lf_vars)])

    combined.attrs.update(
        {
            "case_id": case_id,
            "case_title": case_metadata.title,
            "model_name": model_name,
            "start_date": str(case_metadata.start_date),
            "end_date": str(case_metadata.end_date),
        }
    )

    # ── 6. save ───────────────────────────────────────────────────────────
    output_path = output_dir / model_name / f"case_{case_id:03d}.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # timedelta64 is not natively supported by netcdf4; encode as int nanoseconds
    if "detection_lead_time" in combined:
        combined["detection_lead_time"] = combined[
            "detection_lead_time"
        ].astype("int64")
        combined["detection_lead_time"].attrs["units"] = "nanoseconds"

    combined.to_netcdf(output_path)
    logger.info("Saved to %s", output_path)
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Generate TC track netcdf for the track_map viewer."
    )
    p.add_argument(
        "--case-id",
        type=int,
        required=True,
        help="EWB case_id_number (tropical_cyclone event)",
    )
    p.add_argument(
        "--model",
        required=True,
        choices=ALL_MODELS,
        help="Forecast model name",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available TC case IDs and exit",
    )
    args = p.parse_args()

    if args.list_cases:
        all_cases = ewb.load_cases()
        tc_cases = sorted(
            [c for c in all_cases if c.event_type == "tropical_cyclone"],
            key=lambda c: c.case_id_number,
        )
        print(f"{'ID':>5}  {'Title':<30}  {'Start'}")
        print("-" * 60)
        for c in tc_cases:
            print(
                f"{c.case_id_number:>5}  {c.title:<30}  "
                f"{c.start_date.strftime('%Y-%m-%d')}"
            )
        return

    out = generate(
        case_id=args.case_id,
        model_name=args.model,
        output_dir=args.output_dir,
    )
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
