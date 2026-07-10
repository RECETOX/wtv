"""Ion selection module for wtv package.

This module implements the core ion selection algorithm based on WTV-2.0.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wtv.similarity import (
    calculate_average_score_and_difference_count,
    calculate_combination_score,
    calculate_similarity,
    calculate_solo_compound_combination_score,
    get_ion_list,
)
from wtv.utils import filter_and_sort_combinations, load_data, write_msp


def run_ion_selection(
    msp_file_path: Path,
    output_directory: Path,
    mz_min: float = 0.0,
    mz_max: float = np.inf,
    rt_window: float = 1.0,
    min_ion_intensity_percent: float = 5.0,
    min_ion_num: int = 3,
    prefer_mz_threshold: float = 150.0,
    similarity_threshold: float = 0.85,
    fr_factor: float = 2.0,
    retention_time_max: float = np.inf,
    retention_field: str = "retention_time",
) -> None:
    """Run ion selection on an MSP file.

    Args:
        msp_file_path: Path to input MSP file.
        output_directory: Directory for output files.
        mz_min: Minimum m/z value. Defaults to 0.
        mz_max: Maximum m/z value. Defaults to infinity.
        rt_window: Retention time window for finding nearby compounds. Defaults to 1.0.
        min_ion_intensity_percent: Minimum ion intensity as percentage. Defaults to 5.0.
        min_ion_num: Minimum number of ions required. Defaults to 3.
        prefer_mz_threshold: Preferred m/z threshold for scoring. Defaults to 150.0.
        similarity_threshold: Similarity threshold for identifying similar compounds. Defaults to 0.85.
        fr_factor: Fragment ratio factor. Defaults to 2.0.
        retention_time_max: Maximum retention time/index value. Defaults to infinity.
        retention_field: Field to use for retention data. Defaults to "retention_time".
    """
    logging.info(f"Loading data from {msp_file_path}.")
    RT_data, matrix = load_data(msp_file_path, mz_min, mz_max, retention_field)

    logging.info("Generating ion combinations.")
    combination_result_df = generate_ion_combinations(
        min_ion_intensity_percent=min_ion_intensity_percent,
        min_ion_num=min_ion_num,
        prefer_mz_threshold=prefer_mz_threshold,
        similarity_threshold=similarity_threshold,
        fr_factor=fr_factor,
        RT_data=RT_data,
        matrix=matrix,
        rt_window=rt_window,
    )

    logging.info("Collecting ion retention times.")
    ion_rt = get_ion_rt(
        retention_time_max=retention_time_max,
        RT_data=RT_data,
        combination_result_df=combination_result_df,
    )

    logging.info("Writing MSP file.")
    write_msp(ion_rt, output_directory, msp_file_path)


def get_ion_rt(
    retention_time_max: float,
    RT_data: pd.DataFrame,
    combination_result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract retention times for ion combinations.

    Args:
        retention_time_max: Maximum retention time/index value.
        RT_data: DataFrame with retention data.
        combination_result_df: DataFrame with ion combinations.

    Returns:
        DataFrame with RT and ion information.
    """
    name_list_total: list[str] = []
    num: list[float] = []
    RT_list_total: list[float] = []

    name_list = combination_result_df.index.tolist()

    for name in name_list:
        if name not in RT_data.index:
            logging.error(f"This compound is not in the RT list: {name}")
            continue

        ion_value = combination_result_df.loc[name, "Ion_Combination"]

        if isinstance(ion_value, str):
            ion_list = get_ion_list(ion_value)
        elif isinstance(ion_value, list):
            ion_list = [float(x) for x in ion_value]
        else:
            logging.error(f"The ion group format is incorrect for compound: {name}")
            continue

        for ion in ion_list:
            name_list_total.append(name)
            RT_list_total.append(float(RT_data.loc[name, "RT"]))
            num.append(ion)

    data = {"RT": RT_list_total, "ion": num}
    ion_rt = pd.DataFrame(data, index=name_list_total)

    ion_rt.sort_values(by="RT", inplace=True, ascending=True)

    # Cap retention values at maximum
    mask = ion_rt["RT"] > retention_time_max
    ion_rt.loc[mask, "RT"] = retention_time_max

    return ion_rt


def get_nearby_compounds(
    rt_window: float, RT_data: pd.DataFrame
) -> dict[str, list[str]]:
    """Find compounds within a retention time window.

    Args:
        rt_window: Retention time window size.
        RT_data: DataFrame with retention data.

    Returns:
        Dictionary mapping each compound to its nearby compounds.
    """
    nearby_compound_dic: dict[str, list[str]] = {}

    for name in RT_data.index:
        rt = RT_data.at[name, "RT"]
        nearby = RT_data[
            (RT_data.iloc[:, 0] >= rt - rt_window)
            & (RT_data.iloc[:, 0] <= rt + rt_window)
        ].index.tolist()
        nearby_compound_dic[name] = nearby

    return nearby_compound_dic


def filter_matrix(
    matrix: pd.DataFrame, compound: str, min_ion_intensity: float
) -> pd.DataFrame:
    """Filter matrix to get ions above intensity threshold for a compound.

    Args:
        matrix: Ion intensity matrix.
        compound: Compound name.
        min_ion_intensity: Minimum intensity threshold.

    Returns:
        Filtered DataFrame with ion intensities.
    """
    compound_series = matrix.loc[compound]
    filtered = compound_series[compound_series >= min_ion_intensity]

    if isinstance(filtered, pd.Series):
        filtered = filtered.to_frame(name=compound)
    else:
        filtered = filtered.T

    filtered.dropna(how="all", inplace=True)
    filtered.replace(np.nan, 0, inplace=True)
    filtered["ion"] = filtered.index.astype(float)

    return filtered


def get_ions_for_single_compound(
    RT_data: pd.DataFrame,
    targeted_compound: str,
    matrix: pd.DataFrame,
    min_ion_intensity: float,
    prefer_mz_threshold: float,
    min_ion_num: int,
) -> dict:
    """Get ion combinations for a single compound (no neighbors).

    Args:
        RT_data: DataFrame with retention data.
        targeted_compound: Name of target compound.
        matrix: Ion intensity matrix.
        min_ion_intensity: Minimum intensity threshold.
        prefer_mz_threshold: Preferred m/z threshold.
        min_ion_num: Minimum number of ions.

    Returns:
        Dictionary with ion combination results.
    """
    row: dict = {}
    row["RT"] = RT_data.loc[targeted_compound, "RT"]
    row["Similar_Compound_List"] = []
    row["SCL_Note"] = "No adjacent compounds."

    matrix_1 = filter_matrix(matrix, targeted_compound, min_ion_intensity)

    if matrix_1.shape[0] < 2:
        row["Ion_Combination"] = "NA"
        row["Note"] = (
            "No adjacent compounds; "
            f"The available number of ions ({matrix_1.shape[0]}) is less "
            f"than 2, the compound is excluded"
        )
        return row

    matrix_1 = calculate_solo_compound_combination_score(matrix_1, prefer_mz_threshold)

    if matrix_1.shape[0] <= min_ion_num:
        combination_list = matrix_1.index.tolist()
    else:
        combination_list = matrix_1.iloc[0:min_ion_num].index.tolist()

    row["Ion_Combination"] = combination_list
    return row


def generate_ion_combinations(
    min_ion_intensity_percent: float,
    min_ion_num: int,
    prefer_mz_threshold: float,
    similarity_threshold: float,
    fr_factor: float,
    RT_data: pd.DataFrame,
    matrix: pd.DataFrame,
    rt_window: float,
) -> pd.DataFrame:
    """Generate ion combinations for all compounds.

    Args:
        min_ion_intensity_percent: Minimum ion intensity percentage.
        min_ion_num: Minimum number of ions.
        prefer_mz_threshold: Preferred m/z threshold.
        similarity_threshold: Similarity threshold.
        fr_factor: Fragment ratio factor.
        RT_data: DataFrame with retention data.
        matrix: Ion intensity matrix.
        rt_window: Retention time window.

    Returns:
        DataFrame with ion combinations for all compounds.
    """
    combination_result_df = pd.DataFrame(
        columns=[
            "RT",
            "Ion_Combination",
            "Note",
            "Similar_Compound_List",
            "SCL_Note",
        ]
    )

    nearby_compound_dic = get_nearby_compounds(rt_window, RT_data)
    min_ion_intensity = min_ion_intensity_percent

    def process_compound(args):
        targeted_compound, nearby_compound_list = args

        logging.info(
            f"Processing compound: {targeted_compound} with nearby: {nearby_compound_list}"
        )

        if nearby_compound_list == [targeted_compound]:
            row = get_ions_for_single_compound(
                RT_data=RT_data,
                targeted_compound=targeted_compound,
                matrix=matrix,
                min_ion_intensity=min_ion_intensity,
                prefer_mz_threshold=prefer_mz_threshold,
                min_ion_num=min_ion_num,
            )
        else:
            row = calculate_ion_combination(
                min_ion_num=min_ion_num,
                prefer_mz_threshold=prefer_mz_threshold,
                similarity_threshold=similarity_threshold,
                fr_factor=fr_factor,
                RT_data=RT_data,
                matrix=matrix,
                min_ion_intensity=min_ion_intensity,
                targeted_compound=targeted_compound,
                nearby_compound_list=nearby_compound_list,
            )
        return targeted_compound, row

    # Sequential processing (parallel version commented out for stability)
    results = list(map(process_compound, nearby_compound_dic.items()))

    for targeted_compound, row in results:
        combination_result_df.loc[targeted_compound] = pd.Series(row)

    return combination_result_df


def calculate_ion_combination(
    min_ion_num: int,
    prefer_mz_threshold: float,
    similarity_threshold: float,
    fr_factor: float,
    RT_data: pd.DataFrame,
    matrix: pd.DataFrame,
    min_ion_intensity: float,
    targeted_compound: str,
    nearby_compound_list: list[str],
) -> dict:
    """Calculate optimal ion combination for a compound.

    Args:
        min_ion_num: Minimum number of ions.
        prefer_mz_threshold: Preferred m/z threshold.
        similarity_threshold: Similarity threshold.
        fr_factor: Fragment ratio factor.
        RT_data: DataFrame with retention data.
        matrix: Ion intensity matrix.
        min_ion_intensity: Minimum intensity threshold.
        targeted_compound: Target compound name.
        nearby_compound_list: List of nearby compounds.

    Returns:
        Dictionary with ion combination results.
    """
    row: dict = {}
    row["RT"] = RT_data.loc[targeted_compound, "RT"]

    temp_df = get_nearby_compound_ions(
        matrix=matrix,
        min_ion_intensity=min_ion_intensity,
        targeted_compound=targeted_compound,
        nearby_compound_list=nearby_compound_list,
    )

    if temp_df.shape[1] < 2:
        row["Ion_Combination"] = "NA"
        row["Note"] = (
            "The available number of ions is less than 2, the compound is excluded"
        )
        return row

    similar_compound_list = get_similar_compounds(
        similarity_threshold=similarity_threshold,
        fr_factor=fr_factor,
        targeted_compound=targeted_compound,
        temp_df=temp_df,
    )

    row["Similar_Compound_List"] = similar_compound_list
    temp_df.drop(index=similar_compound_list, inplace=True)

    if temp_df.shape[0] == 1:
        temp_name = temp_df.index.tolist()[0]
        if temp_name == targeted_compound:
            row["SCL_Note"] = np.nan
            ion_combination = get_ions_for_single_compound(
                RT_data=RT_data,
                targeted_compound=targeted_compound,
                matrix=matrix,
                min_ion_intensity=min_ion_intensity,
                prefer_mz_threshold=prefer_mz_threshold,
                min_ion_num=min_ion_num,
            )
            ion_combination.update(row)
            return ion_combination

    col_name = list(temp_df.columns)
    col_name = [float(x) for x in col_name]
    new_com = [[x] for x in col_name]

    difference_count_df_1 = calculate_average_score_and_difference_count(
        targeted_compound=targeted_compound,
        ion_combination=new_com,
        df=temp_df,
        similarity_threshold=similarity_threshold,
        fr_factor=fr_factor,
    )

    combination_df = difference_count_df_1[
        difference_count_df_1["Diff_Count"] >= difference_count_df_1.iat[0, 0]
    ]

    if combination_df.shape[0] > 5:
        combination_df = filter_and_sort_combinations(
            combination_df, "Similar_Compound_Ave_Score"
        )
        if combination_df.shape[0] > 5:
            combination_df = calculate_combination_score(
                combination_df=combination_df,
                targeted_compound=targeted_compound,
                temp_df=temp_df,
                prefer_mz_threshold=prefer_mz_threshold,
            ).sort_values(by="com_score", ascending=False)[:5]

    ion_list = list(temp_df.columns)
    combination_array = combination_df.index.values
    n = 0
    ion_num = 1
    flag = True

    while True:
        max_diff_count = int((combination_df.max()).iloc[0])
        total_ions = int(temp_df.shape[0] - 1)

        if (max_diff_count >= total_ions and ion_num >= min_ion_num) or not flag:
            break

        n += 1
        total_list: list[list[float]] = []
        new_total: list[list[float]] = []

        for ion_combination in combination_array:
            ion_combination_list = get_ion_list(ion_combination)
            candidate_list = [i for i in ion_list if i not in ion_combination_list]

            if not candidate_list:
                if max_diff_count >= total_ions:
                    if combination_df.shape[0] > 1:
                        combination_df = filter_and_sort_combinations(
                            combination_df, "Similar_Compound_Ave_Score"
                        )
                        if combination_df.shape[0] > 1:
                            combination_df = calculate_combination_score(
                                combination_df=combination_df,
                                targeted_compound=targeted_compound,
                                temp_df=temp_df,
                                prefer_mz_threshold=prefer_mz_threshold,
                            ).sort_values(by="com_score", ascending=False)[:1]
                    row["Ion_Combination"] = combination_df.index[0]
                    row["Note"] = (
                        "Despite the qualitative ion number being less than the defined number, "
                        "its separation score reaches 1"
                    )
                    flag = False
                    break
                else:
                    row["Ion_Combination"] = "NA"
                    row["Note"] = "No ions available, this compound is discarded"
                    flag = False
                    break

            if flag:
                for candidate in candidate_list:
                    temp_ion_combination_list = ion_combination_list.copy()
                    temp_ion_combination_list.append(candidate)
                    total_list.append(temp_ion_combination_list)

                seen: set[tuple[float, ...]] = set()
                for item in total_list:
                    item_tuple = tuple(sorted(item))
                    if item_tuple not in seen:
                        seen.add(item_tuple)
                        new_total.append(item)

            if flag:
                difference_count_df_2 = calculate_average_score_and_difference_count(
                    targeted_compound=targeted_compound,
                    ion_combination=new_total,
                    df=temp_df,
                    similarity_threshold=similarity_threshold,
                    fr_factor=fr_factor,
                )

                if len(difference_count_df_2) > 0:
                    combination_df = difference_count_df_2[
                        difference_count_df_2["Diff_Count"]
                        >= difference_count_df_2.iat[0, 0]
                    ]
                else:
                    row["Ion_Combination"] = "NA"
                    row["Note"] = "Error: The difference_count_df is empty."
                    flag = False
                    break

                if combination_df.shape[0] > 1:
                    combination_df = filter_and_sort_combinations(
                        combination_df, "Similar_Compound_Ave_Score"
                    )
                    if combination_df.shape[0] > 1:
                        combination_df = calculate_combination_score(
                            combination_df=combination_df,
                            targeted_compound=targeted_compound,
                            temp_df=temp_df,
                            prefer_mz_threshold=prefer_mz_threshold,
                        ).sort_values(by="com_score", ascending=False)[:1]

                combination_array = combination_df.index.values
                ion_num += 1

        if flag:
            row["Ion_Combination"] = combination_array[0]

    return row


def get_similar_compounds(
    similarity_threshold: float,
    fr_factor: float,
    targeted_compound: str,
    temp_df: pd.DataFrame,
) -> list[str]:
    """Find compounds similar to the target.

    Args:
        similarity_threshold: Minimum similarity score.
        fr_factor: Fragment ratio factor.
        targeted_compound: Target compound name.
        temp_df: DataFrame with spectral data.

    Returns:
        List of similar compound names.
    """
    similar_compound_list: list[str] = []
    result_df = calculate_similarity(targeted_compound, temp_df, fr_factor)

    for index, df_row in result_df.iterrows():
        if float(df_row.iloc[0]) >= similarity_threshold:
            similar_compound_list.append(index)

    return similar_compound_list


def get_nearby_compound_ions(
    matrix: pd.DataFrame,
    min_ion_intensity: float,
    targeted_compound: str,
    nearby_compound_list: list[str],
) -> pd.DataFrame:
    """Get ions from nearby compounds that are also present in target.

    Args:
        matrix: Ion intensity matrix.
        min_ion_intensity: Minimum intensity threshold.
        targeted_compound: Target compound name.
        nearby_compound_list: List of nearby compounds.

    Returns:
        DataFrame with ions from nearby compounds.
    """
    temp_df = matrix.loc[nearby_compound_list]
    temp_df = temp_df.astype(float)

    # Zero out low-intensity ions in target
    temp_df.loc[targeted_compound, :] = np.where(
        temp_df.loc[targeted_compound, :] < min_ion_intensity,
        0,
        temp_df.loc[targeted_compound, :],
    )

    # Keep only columns where target has ions
    temp_df = temp_df.loc[:, temp_df.loc[targeted_compound, :] > 0]

    return temp_df
