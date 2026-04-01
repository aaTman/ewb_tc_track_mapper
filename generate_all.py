"""Batch-generate TC track netcdfs for all cases × all models in parallel.

Usage:
    python generate_all.py                        # all models, all cases
    python generate_all.py --models GRAP_v100_GFS FOUR_v200_GFS
    python generate_all.py --workers 4
    python generate_all.py --skip-existing        # skip already-generated files
"""

import argparse
import logging
import pathlib
import time
import traceback

from joblib import Parallel, delayed

import extremeweatherbench as ewb
from extremeweatherbench.inputs import CIRA_MODEL_NAMES

from generate import ALL_MODELS, DEFAULT_OUTPUT_DIR, generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(processName)s  %(message)s",
)
logger = logging.getLogger(__name__)


_RETRYABLE = ("timeout", "timed out", "connect", "connection", "icechunkerror")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in _RETRYABLE)


def _safe_generate(
    case_id: int,
    model: str,
    output_dir: pathlib.Path,
    skip_existing: bool,
    retries: int = 3,
    retry_delay: float = 10.0,
) -> dict:
    """Wrapper around generate() that catches exceptions so one failure
    doesn't abort the whole batch. Retries on transient network errors."""
    out_path = output_dir / model / f"case_{case_id:03d}.nc"
    if skip_existing and out_path.exists():
        logger.info("Skipping %s / case %d (already exists)", model, case_id)
        return {"case_id": case_id, "model": model, "status": "skipped"}

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            generate(case_id=case_id, model_name=model, output_dir=output_dir)
            return {"case_id": case_id, "model": model, "status": "ok"}
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < retries:
                wait = retry_delay * attempt
                logger.warning(
                    "Network error on case %d / %s (attempt %d/%d), "
                    "retrying in %.0fs: %s",
                    case_id, model, attempt, retries, wait, exc,
                )
                time.sleep(wait)
            else:
                break

    logger.error(
        "FAILED case %d / %s: %s\n%s",
        case_id,
        model,
        last_exc,
        traceback.format_exc(),
    )
    return {"case_id": case_id, "model": model, "status": "error", "error": str(last_exc)}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate all TC track netcdfs in parallel."
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        metavar="MODEL",
        help="Models to generate (default: all). Choices: %(choices)s",
    )
    p.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Specific case IDs to generate (default: all TC cases)",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8; use -1 for all CPUs)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries on transient network errors (default: 3)",
    )
    p.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Base delay in seconds between retries, multiplied by attempt "
             "number (default: 10)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases whose netcdf already exists",
    )
    args = p.parse_args()

    all_cases = ewb.load_cases()
    tc_cases = sorted(
        [c for c in all_cases if c.event_type == "tropical_cyclone"],
        key=lambda c: c.case_id_number,
    )

    if args.case_ids:
        tc_cases = [c for c in tc_cases if c.case_id_number in args.case_ids]
        if not tc_cases:
            raise SystemExit(f"No TC cases found for IDs: {args.case_ids}")

    jobs = [
        (c.case_id_number, model)
        for c in tc_cases
        for model in args.models
    ]

    logger.info(
        "Generating %d jobs (%d cases × %d models) with %s workers",
        len(jobs),
        len(tc_cases),
        len(args.models),
        args.workers if args.workers != -1 else "all CPU",
    )

    results = Parallel(n_jobs=args.workers, backend="loky", verbose=10)(
        delayed(_safe_generate)(
            case_id, model, args.output_dir, args.skip_existing,
            args.retries, args.retry_delay,
        )
        for case_id, model in jobs
    )

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'─' * 50}")
    print(f"  Done:    {len(ok)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Errors:  {len(errors)}")
    if errors:
        print("\nFailed jobs:")
        for e in errors:
            print(f"  case {e['case_id']:>4d} / {e['model']}: {e['error']}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
