"""Command-line interface for wtv package.

This module provides the CLI for ion selection in mass spectrometry data.
"""

import argparse
import logging
import sys
from pathlib import Path

from matchms.exporting import save_spectra
from matchms.importing import load_spectra

from wtv.ion_selection import generate_ion_combinations, run_ion_selection
from wtv.utils import (
    RETENTION_FIELDS,
    RT_INDEX_WINDOW_DEFAULT,
    RT_WINDOW_DEFAULT,
    MissingRetentionDataError,
    get_filtered_spectra,
    parse_spectra,
    resolve_rt_window,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate methods for compound analysis."
    )
    parser.add_argument(
        "--msp_path", type=str, required=True, help="Path to the MSP file."
    )
    parser.add_argument(
        "--outpath", type=str, required=True, help="Output path for results."
    )
    parser.add_argument(
        "--mz_min",
        type=float,
        required=False,
        help="Minimum m/z value.",
        default=35.0,
    )
    parser.add_argument(
        "--mz_max",
        type=float,
        required=False,
        help="Maximum m/z value.",
        default=400.0,
    )
    parser.add_argument(
        "--rt_window",
        type=float,
        required=False,
        help=(
            "Retention window for finding nearby compounds, in the units of "
            "--retention_field. Must be comparable to the distance between "
            "co-eluting compounds, not to the length of the gradient. Defaults "
            f"to {RT_WINDOW_DEFAULT} (minutes) for retention_time and to "
            f"{RT_INDEX_WINDOW_DEFAULT} (index units) for retention_index."
        ),
        default=None,
    )
    parser.add_argument(
        "--min_ion_intensity_percent",
        type=float,
        required=False,
        help="Minimum ion intensity percent.",
        default=7.0,
    )
    parser.add_argument(
        "--min_ion_num",
        type=int,
        required=False,
        help="Minimum number of ions.",
        default=2,
    )
    parser.add_argument(
        "--prefer_mz_threshold",
        type=float,
        required=False,
        help="Preferred m/z threshold.",
        default=60.0,
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        required=False,
        help="Similarity threshold.",
        default=0.85,
    )
    parser.add_argument(
        "--fr_factor",
        type=float,
        required=False,
        help="FR factor for weighted dot product calculation.",
        default=2.0,
    )
    parser.add_argument(
        "--retention_time_max",
        type=float,
        required=False,
        help="Maximum retention time or index value.",
        default=float("inf"),
    )
    parser.add_argument(
        "--retention_field",
        type=str,
        required=False,
        help=(
            "Field to use for retention data. GC-MS libraries often carry a "
            "'retention_index' instead of a 'retention_time'; select whichever the "
            "input file provides. Execution stops with an error when the selected "
            "field carries no usable values."
        ),
        default="retention_time",
        choices=list(RETENTION_FIELDS),
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for wtv CLI."""
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Parsed arguments successfully.")

    rt_window = resolve_rt_window(args.rt_window, args.retention_field)

    try:
        run_ion_selection(
            msp_file_path=Path(args.msp_path),
            output_directory=Path(args.outpath),
            mz_min=args.mz_min,
            mz_max=args.mz_max,
            rt_window=rt_window,
            min_ion_intensity_percent=args.min_ion_intensity_percent,
            min_ion_num=args.min_ion_num,
            prefer_mz_threshold=args.prefer_mz_threshold,
            similarity_threshold=args.similarity_threshold,
            fr_factor=args.fr_factor,
            retention_time_max=args.retention_time_max,
            retention_field=args.retention_field,
        )
    except MissingRetentionDataError as error:
        # Stop with the guidance alone - a traceback adds nothing for the user here.
        logging.error("%s", error)
        sys.exit(1)


def main_v2() -> None:
    """Alternative entry point using matchms spectra directly."""
    args = parse_args()

    rt_window = resolve_rt_window(args.rt_window, args.retention_field)

    spectra = list(load_spectra(args.msp_path))
    matrix, rts = parse_spectra(
        spectra,
        retention_field=args.retention_field,
        mz_min=args.mz_min,
        mz_max=args.mz_max,
    )

    combination_result_df = generate_ion_combinations(
        min_ion_intensity_percent=args.min_ion_intensity_percent,
        min_ion_num=args.min_ion_num,
        prefer_mz_threshold=args.prefer_mz_threshold,
        similarity_threshold=args.similarity_threshold,
        fr_factor=args.fr_factor,
        RT_data=rts,
        matrix=matrix,
        rt_window=rt_window,
    )

    results = get_filtered_spectra(spectra, combination_result_df)
    save_spectra(results, args.outpath)


if __name__ == "__main__":
    main()
