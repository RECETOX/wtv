# This file is a derivative work based on code by Honglun Yuan et al.
# Original repository: https://github.com/yuanhonglun/WTV_2.0
# Original publication: https://doi.org/10.1016/j.molp.2024.04.012


import re
from pathlib import Path

import logging

# logger = logging.getLogger(__name__)

import concurrent
import numpy as np
import pandas as pd

from wtv.similarity import (
    calculate_average_score_and_difference_count,
    calculate_combination_score,
    calculate_similarity,
    calculate_solo_compound_combination_score,
)
from wtv.utils import (
    average_rts_for_duplicated_indices,
    check_rt_data,
    create_ion_matrix,
    filter_and_sort_combinations,
    read_msp,
    write_msp,
)


def run_ion_selection(
    msp_file_path: Path,
    output_directory: Path,
    mz_min,
    mz_max,
    rt_window,
    min_ion_intensity_percent,
    min_ion_num,
    prefer_mz_threshold,
    similarity_threshold,
    fr_factor,
    retention_time_max,
):
    logging.info(f"Loading data from file at {msp_file_path}.")
    RT_data, matrix = load_data(msp_file_path, mz_min, mz_max)

    logging.info("Generating ion combinations.")
    combination_result_df = generate_ion_combinations(
        min_ion_intensity_percent,
        min_ion_num,
        prefer_mz_threshold,
        similarity_threshold,
        fr_factor,
        RT_data,
        matrix,
        rt_window,
    )

    ion_rt = get_ion_rt(retention_time_max, RT_data, combination_result_df)

    write_msp(ion_rt, output_directory, msp_file_path)


def load_data(msp_file_path, mz_min, mz_max):
    meta_1, RT_data = read_msp(msp_file_path)
    matrix = create_ion_matrix(mz_min, mz_max, meta_1)
    RT_data = average_rts_for_duplicated_indices(RT_data)
    check_rt_data(RT_data)
    RT_data = RT_data.sort_values(by="RT")
    return RT_data, matrix


def get_ion_rt(retention_time_max, RT_data, combination_result_df):
    name_list_total = []
    num = []
    name_list = combination_result_df.index.values.tolist()
    RT_list_total = []
    for name in name_list:
        if name in RT_data.index.values.tolist():
            if isinstance(
                combination_result_df.loc[name, "Ion_Combination"], str
            ):  # Fixed type check
                ion_str = combination_result_df.loc[name, "Ion_Combination"]
                ion_list = re.findall(r"\d+\.?\d*", ion_str)

                ion_list = list(map(float, ion_list))
                for x in range(0, len(ion_list)):
                    name_list_total.append(name)
                    RT_list_total.append(RT_data.loc[name, "RT"])
                    num.append(ion_list[x])
            elif isinstance(
                combination_result_df.loc[name, "Ion_Combination"], list
            ):  # Fixed type check
                ion_list = [
                    int(x) for x in combination_result_df.loc[name, "Ion_Combination"]
                ]
                for x in range(0, len(ion_list)):
                    name_list_total.append(name)
                    RT_list_total.append(RT_data.loc[name, "RT"])
                    num.append(ion_list[x])
            else:
                logging.error(f"The ion group format is incorrect for compound: {name}")
        else:
            logging.error(f"This compound is not in the RT list: {name}")

    data = {"RT": RT_list_total, "ion": num}

    ion_rt = pd.DataFrame(data, index=name_list_total)

    ion_rt.sort_values(by="RT", inplace=True, ascending=True)
    for idx, row in ion_rt.iterrows():
        if row["RT"] > retention_time_max:
            ion_rt.loc[idx, "RT"] = retention_time_max
    return ion_rt


def get_nearby_compounds(rt_window, RT_data):
    nearby_compound_dic = {}
    for name in RT_data.index.values.tolist():
        rt = RT_data.at[name, "RT"]
        nearby_compound_dic[name] = RT_data[
            (RT_data.iloc[:, 0] >= rt - rt_window)
            & (RT_data.iloc[:, 0] <= rt + rt_window)
        ].index.tolist()

    return nearby_compound_dic


def filter_matrix(
    matrix: pd.DataFrame, compound: str, min_ion_intensity: float
) -> pd.DataFrame:
    """
    Return a DataFrame of ions for the given compound with intensity above the threshold.
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
    RT_data,
    targeted_compound,
    matrix,
    min_ion_intensity,
    prefer_mz_threshold,
    min_ion_num,
):
    row: dict = {}
    row["RT"] = RT_data.loc[targeted_compound, "RT"]
    row["Similar_Compound_List"] = []
    row["SCL_Note"] = "No adjacent compounds."

    matrix_1 = filter_matrix(matrix, targeted_compound, min_ion_intensity)

    if matrix_1.shape[0] < 2:
        row["Ion_Combination"] = "NA"
        row["Note"] = (
            "No adjacent compounds; "
            "The available number of ions is less "
            "than 2, the compound is excluded"
        )
        return row

    matrix_1 = calculate_solo_compound_combination_score(matrix_1, prefer_mz_threshold)

    if matrix_1.shape[0] <= min_ion_num:
        combination_list = matrix_1.index.values.tolist()
    else:
        combination_list = matrix_1.iloc[0:min_ion_num, :].index.values.tolist()
    row["Ion_Combination"] = combination_list
    return row


def generate_ion_combinations(
    min_ion_intensity_percent,
    min_ion_num,
    prefer_mz_threshold,
    similarity_threshold,
    fr_factor,
    RT_data,
    matrix,
    rt_window,
):
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

        logging.info(f"Processing compound: {targeted_compound} with nearby compounds: {nearby_compound_list}")

        if nearby_compound_list == [targeted_compound]:
            row = get_ions_for_single_compound(
                RT_data,
                targeted_compound,
                matrix,
                min_ion_intensity,
                prefer_mz_threshold,
                min_ion_num,
            )
        else:
            row = calculate_ion_combination(
                min_ion_num,
                prefer_mz_threshold,
                similarity_threshold,
                fr_factor,
                RT_data,
                matrix,
                min_ion_intensity,
                targeted_compound,
                nearby_compound_list,
            )
        return targeted_compound, row

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(process_compound, nearby_compound_dic.items()))

    for targeted_compound, row in results:
        combination_result_df.loc[targeted_compound] = pd.Series(row)
    return combination_result_df

def calculate_ion_combination(
    min_ion_num,
    prefer_mz_threshold,
    similarity_threshold,
    fr_factor,
    RT_data,
    matrix,
    min_ion_intensity,
    targeted_compound,
    nearby_compound_list,
):
    row: dict = {}
    row["RT"] = RT_data.loc[targeted_compound, "RT"]

    temp_df = get_nearby_compound_ions(
        matrix, min_ion_intensity, targeted_compound, nearby_compound_list
    )

    if temp_df.shape[1] < 2:
        row["Ion_Combination"] = "NA"
        row["Note"] = (
            "The available number of ions is less " "than 2, the compound is excluded"
        )
        return row

    similar_compound_list = get_similar_compounds(similarity_threshold, fr_factor, targeted_compound, temp_df)

    row["Similar_Compound_List"] = similar_compound_list
    temp_df.drop(index=similar_compound_list, inplace=True)

    if temp_df.shape[0] == 1:
        temp_name = ((temp_df.index.values).tolist())[0]
        if temp_name == targeted_compound:
            row["SCL_Note"] = np.nan
            ion_combination = get_ions_for_single_compound(
                RT_data,
                targeted_compound,
                matrix,
                min_ion_intensity,
                prefer_mz_threshold,
                min_ion_num,
            )
            ion_combination.update(row)
            return ion_combination

    col_name = list(temp_df.columns)

    col_name = list(map(float, col_name))
    new_com = [[x] for x in col_name]

    difference_count_df_1 = calculate_average_score_and_difference_count(
        targeted_compound,
        new_com,
        temp_df,
        similarity_threshold,
        fr_factor,
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
                combination_df,
                targeted_compound,
                temp_df,
                prefer_mz_threshold,
            ).sort_values(by="com_score", inplace=False, ascending=False)[:5]

    ion_list = list(temp_df)
    combination_array = combination_df.index.values
    n = 0
    ion_num = 1

    flag = True

    while True:
        if (int((combination_df.max()).iloc[0]) >= int(temp_df.shape[0] - 1) and ion_num >= min_ion_num) or not flag:
            break

        n = n + 1
        total_list = []
        new_total = []

        for ion_combination in combination_array:
            ion_combination_list = re.findall(r"\d+\.?\d*", ion_combination)
            ion_combination_list = list(map(float, ion_combination_list))
            candidate_list = [i for i in ion_list if i not in ion_combination_list]

            if candidate_list == []:
                if int((combination_df.max())[0]) >= int(temp_df.shape[0] - 1):
                    if combination_df.shape[0] > 1:
                        combination_df = filter_and_sort_combinations(
                            combination_df,
                            "Similar_Compound_Ave_Score",
                        )
                        if combination_df.shape[0] > 1:
                            combination_df = calculate_combination_score(
                                combination_df,
                                targeted_compound,
                                temp_df,
                                prefer_mz_threshold,
                            )
                            combination_df = combination_df.sort_values(
                                by="com_score",
                                inplace=False,
                                ascending=False,
                            )
                            combination_df = combination_df[:1]
                        else:
                            combination_df = combination_df
                    else:
                        combination_df = combination_df
                    combination_array = combination_df.index.values
                    row["Ion_Combination"] = combination_array[0]
                    row["Note"] = (
                        "Despite the qualitative ion number is less than the defined number, its separation score reaches 1"
                    )
                    flag = False
                    break

                else:
                    row["Ion_Combination"] = "NA"
                    row["Note"] = "No ions available, this compound is discarded"
                    flag = False
                    break

            elif flag:  # Fixed equality comparison
                for candidate in candidate_list:
                    temp_ion_combination_list = ion_combination_list.copy()
                    temp_ion_combination_list.append(candidate)
                    total_list.append(temp_ion_combination_list)
                    temp_ion_combination_list = []

                for i in total_list:
                    i = list(map(float, i))
                    i.sort()
                    if i not in new_total:
                        new_total.append(i)
            if flag:  # Fixed equality comparison
                difference_count_df_2 = (
                    calculate_average_score_and_difference_count(
                        targeted_compound,
                        new_total,
                        temp_df,
                        similarity_threshold,
                        fr_factor,
                    )
                )

                if len(difference_count_df_2) > 0:
                    combination_df = difference_count_df_2[
                        difference_count_df_2["Diff_Count"]
                        >= difference_count_df_2.iat[0, 0]
                    ]
                else:
                    row["Ion_Combination"] = "NA"
                    row["Note"] = "Error: The 'difference_count_df' is " "empty."
                    flag = False

                    break

                if combination_df.shape[0] > 1:
                    combination_df = filter_and_sort_combinations(
                        combination_df, "Similar_Compound_Ave_Score"
                    )
                    if combination_df.shape[0] > 1:
                        combination_df = calculate_combination_score(
                            combination_df,
                            targeted_compound,
                            temp_df,
                            prefer_mz_threshold,
                        )
                        combination_df = combination_df.sort_values(
                            by="com_score",
                            inplace=False,
                            ascending=False,
                        )
                        combination_df = combination_df[:1]
                    else:
                        combination_df = combination_df
                else:
                    combination_df = combination_df
                combination_array = combination_df.index.values
                ion_num = ion_num + 1
            else:
                break
        if flag:  # Fixed equality comparison
            row["Ion_Combination"] = combination_array[0]
    return row

def get_similar_compounds(similarity_threshold, fr_factor, targeted_compound, temp_df):
    similar_compound_list = []
    result_df_1 = calculate_similarity(targeted_compound, temp_df, fr_factor)
    for index, df_row in result_df_1.iterrows():
        if float(df_row.iloc[0]) >= similarity_threshold:
            similar_compound_list.append(index)
    return similar_compound_list


def get_nearby_compound_ions(
    matrix, min_ion_intensity, targeted_compound, nearby_compound_list
):
    temp_df = matrix.loc[nearby_compound_list]
    temp_df = temp_df.astype(float)
    temp_df.loc[targeted_compound, :] = np.where(
        temp_df.loc[targeted_compound, :] < min_ion_intensity,
        0,
        temp_df.loc[targeted_compound, :],
    )
    temp_df = temp_df.loc[:, temp_df.loc[targeted_compound, :] > 0]
    return temp_df
