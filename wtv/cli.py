import argparse
from pathlib import Path
from wtv.ion_selection import main as run_ion_selection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate methods for compound analysis."
    )
    parser.add_argument(
        "--msp_path", 
        type=str, 
        required=True, 
        help="Path to the MSP file."
    )
    parser.add_argument(
        "--outpath", 
        type=str, 
        required=True, 
        help="Output path for results."
    )
    parser.add_argument(
        "--mz_min", 
        type=int, 
        required=True, 
        help="Minimum m/z value.", 
        default=35
    )
    parser.add_argument(
        "--mz_max", 
        type=int, 
        required=True, 
        help="Maximum m/z value.", 
        default=400
    )
    parser.add_argument(
        "--rt_window", 
        type=float, 
        required=True, 
        help="RT window value.", 
        default=2.00
    )
    parser.add_argument(
        "--min_ion_intensity_percent",
        type=float,
        required=True,
        help="Minimum ion intensity percent.",
        default=7,
    )
    parser.add_argument(
        "--min_ion_num",
        type=int,
        required=True,
        help="Minimum number of ions.",
        default=2,
    )
    parser.add_argument(
        "--prefer_mz_threshold",
        type=int,
        required=True,
        help="Preferred m/z threshold.",
        default=60,
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        required=True,
        help="Similarity threshold.",
        default=0.85,
    )
    parser.add_argument(
        "--fr_factor", type=float, required=True, help="FR factor.", default=2.0
    )
    parser.add_argument(
        "--retention_time_max",
        type=float,
        required=True,
        help="Maximum retention time.",
        default=68.80,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_ion_selection(
        msp_file_path=Path(args.msp_path),
        output_directory=Path(args.outpath),
        mz_min=args.mz_min,
        mz_max=args.mz_max,
        rt_window=args.rt_window,
        min_ion_intensity_percent=args.min_ion_intensity_percent,
        min_ion_num=args.min_ion_num,
        prefer_mz_threshold=args.prefer_mz_threshold,
        similarity_threshold=args.similarity_threshold,
        fr_factor=args.fr_factor,
        retention_time_max=args.retention_time_max,
    )


if __name__ == "__main__":
    main()
