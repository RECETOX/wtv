# This file is a derivative work based on code by Honglun Yuan et al.
# Original repository: https://github.com/yuanhonglun/WTV_2.0
# Original publication: https://doi.org/10.1016/j.molp.2024.04.012

import re
from pathlib import Path

import numpy as np
import pandas as pd

from wtv.utils import read_msp, write_msp


def dot_product_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the dot product distance between two vectors p and q.

    Args:
        p (numpy.ndarray): First vector.
        q (numpy.ndarray): Second vector.

    Returns:
        float: Dot product distance between the two vectors.
    """
    if np.sum(p) == 0 or np.sum(q) == 0:
        return 0
    return np.power(np.sum(q * p), 2) / (
        np.sum(np.power(q, 2)) * np.sum(np.power(p, 2))
    )


def weighted_dot_product_distance(compare_df: pd.DataFrame, fr_factor: float) -> float:
    """
    Calculate the weighted dot product distance between two vectors in a DataFrame.

    Args:
        compare_df (pd.DataFrame): DataFrame with two columns representing the vectors to compare.
        fr_factor (float): Factor used in the calculation.

    Returns:
        float: Composite score based on the weighted dot product distance.
    """
    m_q = pd.Series(compare_df.index)
    m_q = m_q.astype(float)
    i_q = np.array(compare_df.iloc[:, 0])
    i_r = np.array(compare_df.iloc[:, 1])
    k = 0.5
    exponent = 2  # Renamed from `l` to `exponent`
    w_q = np.power(i_q, k) * np.power(m_q, exponent)
    w_r = np.power(i_r, k) * np.power(m_q, exponent)
    ss = dot_product_distance(w_q, w_r)
    shared_spec = np.vstack((i_q, i_r))
    shared_spec = pd.DataFrame(shared_spec)
    shared_spec = shared_spec.loc[:, (shared_spec != 0).all(axis=0)]
    m = int(shared_spec.shape[1])
    if m >= fr_factor:
        FR = 0
        for i in range(1, m):
            s = (shared_spec.iat[0, i] / shared_spec.iat[0, (i - 1)]) * (
                shared_spec.iat[1, (i - 1)] / shared_spec.iat[1, i]
            )
            if s > 1:
                s = 1 / s
            FR = FR + s
        ave_FR = FR / (m - 1)
        NU = int(len(compare_df))
        composite_score = ((NU * ss) + (m * ave_FR)) / (NU + m)
    else:
        composite_score = ss

    return composite_score


def calculate_similarity(
    target_name: str, df: pd.DataFrame, fr_factor: float
) -> pd.DataFrame:
    """
    Calculate the similarity scores between the target compound and all compounds in the DataFrame.

    Args:
        target_name (str): The name of the target compound.
        df (pd.DataFrame): The DataFrame containing compound data.
        fr_factor (float): The factor used in the weighted dot product distance calculation.

    Returns:
        pd.DataFrame: A DataFrame containing similarity scores for each compound.

    """
    result_df = pd.DataFrame(columns=["Score"])
    first_col = df.loc[target_name]
    for compound in df.index.values:
        if compound != target_name:
            second_col = df.loc[compound]
            compare_df = pd.concat([first_col, second_col], axis=1)
            compare_df = compare_df.astype(float)
            score = weighted_dot_product_distance(compare_df, fr_factor)
            result_df.loc[compound, "Score"] = score

    return result_df


def calculate_average_score_and_difference_count(
    targeted_compound: str,
    ion_combination: list,
    df: pd.DataFrame,
    similarity_threshold: float,
    fr_factor: float,
) -> pd.DataFrame:
    """
    Calculate the average similarity score and the difference count for a targeted compound using specified ion combinations.

    Args:
        targeted_compound (str): The name of the targeted compound.
        ion_combination (list): List of ion combinations for similarity calculation.
        df (pd.DataFrame): The DataFrame containing compound data.
        similarity_threshold (float): The similarity threshold for considering compounds as similar.
        fr_factor (float): The factor used in the weighted dot product distance calculation.

    Returns:
        pd.DataFrame: A DataFrame containing difference counts and average similarity scores for each ion combination.

    """
    difference_count_df = pd.DataFrame(
        columns=["Diff_Count", "Similar_Compound_Ave_Score"]
    )

    for ions in ion_combination:

        temp_df_1 = df[ions]
        result_df_2 = calculate_similarity(targeted_compound, temp_df_1, fr_factor)
        result_df_3 = result_df_2[(result_df_2["Score"] < similarity_threshold)]
        count = len(result_df_3)
        difference_count_df.loc[str(ions), "Diff_Count"] = count

        result_df_4 = result_df_2[(result_df_2["Score"] >= similarity_threshold)]
        if result_df_4.shape[0] > 0:
            ave_score_1 = np.average(result_df_4, axis=0)[0]
        else:
            ave_score_1 = 1
        difference_count_df.loc[str(ions), "Similar_Compound_Ave_Score"] = ave_score_1

    difference_count_df.sort_values(by="Diff_Count", inplace=True, ascending=False)
    return difference_count_df


def calculate_combination_score(
    combination_df, targeted_compound, temp_df, prefer_mz_threshold
):
    """
    Calculate the combination score for ion combinations in a DataFrame.

    Args:
        combination_df (pd.DataFrame): DataFrame containing ion combinations.
        targeted_compound (str): The name of the targeted compound.
        temp_df (pd.DataFrame): Temporary DataFrame containing compound data.
        prefer_mz_threshold (int): The preferred m/z threshold.

    Returns:
        pd.DataFrame: DataFrame with combination scores added.

    """
    for index, _ in combination_df.iterrows():
        ion_list = re.findall(r"\d+\.?\d*", index)
        ion_list = list(map(float, ion_list))
        new_temp_df = temp_df.loc[str(targeted_compound), ion_list].to_frame()
        new_temp_df["ion"] = new_temp_df.index.tolist()
        new_temp_df["ion"] = new_temp_df["ion"].astype("int")
        new_temp_df["ion"] = np.where(
            new_temp_df["ion"] < prefer_mz_threshold, 1, new_temp_df["ion"]
        )
        new_temp_df["score"] = (pow(new_temp_df["ion"], 3)) * (
            pow(new_temp_df[str(targeted_compound)], 0.5)
        )
        combination_df.loc[index, "com_score"] = new_temp_df["score"].sum()

    return combination_df


def calculate_solo_compound_combination_score(matrix_1, prefer_mz_threshold):
    """
    Calculate the combination score for solo compounds in a DataFrame.

    Args:
        matrix_1 (pd.DataFrame): DataFrame containing solo compound data.
        prefer_mz_threshold (int): The preferred m/z threshold.

    Returns:
        pd.DataFrame: DataFrame with combination scores added and sorted by score.

    """
    matrix_1["ion"] = matrix_1["ion"].apply(
        lambda x: 1 if x < prefer_mz_threshold else x
    )
    matrix_1["com_score"] = matrix_1.apply(
        lambda row: pow(row.iloc[0], 0.5) * pow(row.iloc[1], 3), axis=1
    )
    matrix_1 = matrix_1.sort_values(by="com_score", ascending=False)
    return matrix_1


def filter_matrix(
    matrix: pd.DataFrame, compound: str, min_ion_intensity: float
) -> pd.DataFrame:
    """
    Filter the matrix for a specific compound based on minimum ion intensity.

    Args:
        matrix (pd.DataFrame): The DataFrame containing compound data.
        compound (str): The name of the compound to filter.
        min_ion_intensity (float): The minimum ion intensity threshold.

    Returns:
        pd.DataFrame: A filtered DataFrame containing ions with intensity above the threshold.
    """
    # Extract the compound data as a DataFrame
    matrix_1 = matrix.loc[compound].to_frame()

    # Add a column for ion values converted to integers
    matrix_1["ion"] = matrix_1.index.astype(int)

    # Apply the minimum ion intensity threshold
    matrix_1[compound] = np.where(
        matrix_1[compound].astype(float) < min_ion_intensity,
        0,
        matrix_1[compound].astype(float),
    )

    # Return the filtered DataFrame with ions above the threshold
    return matrix_1.loc[matrix_1[compound] > 0, :]


def filter_and_sort_combinations(
    combination_df: pd.DataFrame, score_column: str
) -> pd.DataFrame:
    """
    Filter and sort combinations in a DataFrame based on a score column.

    Args:
        combination_df (pd.DataFrame): DataFrame containing combinations and their scores.
        score_column (str): The name of the column containing scores to filter and sort by.

    Returns:
        pd.DataFrame: A filtered and sorted DataFrame based on the score column.
    """
    # Sort the DataFrame by the specified score column in ascending order
    combination_df = combination_df.sort_values(
        by=score_column, inplace=False, ascending=True
    )

    # Filter the DataFrame to include only rows with scores greater than or equal to the minimum score
    return combination_df[combination_df[score_column] >= combination_df.iat[0, 1]]


def main(
    msp_file_path,
    output_directory,
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
    # Convert paths to Path objects
    msp_file_path = Path(msp_file_path)
    output_directory = Path(output_directory)

    meta_1, RT_data = read_msp(msp_file_path)

    error_df = pd.DataFrame(columns=["error"])

    matrix = pd.DataFrame()
    for name, dic in meta_1.items():
        b = {name: dic}
        y = pd.DataFrame.from_dict(b).T
        matrix = pd.concat([matrix, y], axis=0, join="outer").fillna(0)

    for col_name in list(matrix):
        if (
            type(col_name) not in [float, int, np.float64, np.int64]
            and not col_name.isdigit()  # Fixed equality comparison
        ):
            matrix.drop(columns=col_name, inplace=True)

    for ion in list(matrix):
        if int(ion) < mz_min or int(ion) > mz_max:
            matrix.drop(columns=ion, inplace=(True))

    matrix_name = matrix.index.tolist()

    duplicated_index = RT_data.index[RT_data.index.duplicated()]
    for index in duplicated_index:
        error_df.loc[index, "error"] = "Duplicated RT"
    RT_data.drop(duplicated_index, axis=0, inplace=True)
    for index_1, row in RT_data.iterrows():

        if index_1 not in matrix_name:
            error_df.loc[index_1, "error"] = (
                "This compound is in RT list, but not in MSP library"
            )
            RT_data.drop(index=index_1, inplace=True)

        elif (
            type(row.iloc[0]) not in [float, int, np.float64, np.int64]
            or row.iloc[0] == np.nan
        ):
            error_df.loc[index_1, "error"] = "RT format error"
            RT_data.drop(index=index_1, inplace=True)

    RT_data = RT_data.sort_values(by="RT")

    compound_list = RT_data.index.values.tolist()

    # TO DO - Delete this line and all that goes with it
    error_df.to_csv(
        output_directory / "input_data_error_info.csv",
        index=True,
        index_label="Name",
    )
    nearby_compound_dic = {}
    for name in compound_list:
        if name in RT_data.index.values.tolist():
            rt = RT_data.at[name, "RT"]
            nearby_compound_dic[name] = RT_data[
                (RT_data.iloc[:, 0] >= rt - rt_window)
                & (RT_data.iloc[:, 0] <= rt + rt_window)
            ].index.tolist()

    combination_result_df = pd.DataFrame(
        columns=[
            "RT",
            "Ion_Combination",
            "Note",
            "Similar_Compound_List",
            "SCL_Note",
        ]
    )

    min_ion_intensity = min_ion_intensity_percent

    for targeted_compound, nearby_compound_list in nearby_compound_dic.items():
        combination_result_df.loc[targeted_compound, "RT"] = RT_data.loc[
            targeted_compound, "RT"
        ]
        solo_list = []
        if nearby_compound_list == [targeted_compound]:
            scl = []
            combination_result_df.loc[targeted_compound, "Similar_Compound_List"] = scl
            combination_result_df.loc[targeted_compound, "SCL_Note"] = (
                "No adjacent compounds."
            )

            matrix_1 = filter_matrix(matrix, targeted_compound, min_ion_intensity)

            if matrix_1.shape[0] < 2:
                combination_result_df.loc[targeted_compound, "Ion_Combination"] = "NA"
                combination_result_df.loc[targeted_compound, "Note"] = (
                    "No adjacent compounds; "
                    "The available number of ions is less "
                    "than 2, the compound is excluded"
                )
            else:
                matrix_1 = calculate_solo_compound_combination_score(
                    matrix_1, prefer_mz_threshold
                )

                if matrix_1.shape[0] <= min_ion_num:
                    combination_list = matrix_1.index.values.tolist()
                else:
                    combination_list = matrix_1.iloc[
                        0:min_ion_num, :
                    ].index.values.tolist()
                combination_result_df.loc[str(targeted_compound), "Ion_Combination"] = (
                    combination_list
                )
                solo_list.append(targeted_compound)
        else:
            temp_df = matrix.loc[nearby_compound_list]
            temp_df = temp_df.astype(float)
            temp_df.loc[targeted_compound, :] = np.where(
                temp_df.loc[targeted_compound, :] < min_ion_intensity,
                0,
                temp_df.loc[targeted_compound, :],
            )
            temp_df = temp_df.loc[:, temp_df.loc[targeted_compound, :] > 0]

            if temp_df.shape[1] < 2:
                combination_result_df.loc[targeted_compound, "Ion_Combination"] = "NA"
                combination_result_df.loc[targeted_compound, "Note"] = (
                    "The available number of ions is less "
                    "than 2, the compound is excluded"
                )
            else:
                similar_compound_list = []
                result_df_1 = calculate_similarity(
                    targeted_compound, temp_df, fr_factor
                )
                for index, row in result_df_1.iterrows():
                    if float(row.iloc[0]) >= similarity_threshold:
                        similar_compound_list.append(index)

                combination_result_df.loc[
                    targeted_compound, "Similar_Compound_List"
                ] = similar_compound_list
                temp_df.drop(index=similar_compound_list, inplace=True)

                if temp_df.shape[0] == 1:
                    temp_name = ((temp_df.index.values).tolist())[0]
                    if temp_name == targeted_compound:
                        matrix_1 = filter_matrix(
                            matrix, targeted_compound, min_ion_intensity
                        )

                        if matrix_1.shape[0] < 2:
                            combination_result_df.loc[
                                targeted_compound, "Ion_Combination"
                            ] = "NA"
                            combination_result_df.loc[targeted_compound, "Note"] = (
                                "The available number of ions is less "
                                "than 2, the compound is discarded"
                            )
                        else:
                            matrix_1 = calculate_solo_compound_combination_score(
                                matrix_1, prefer_mz_threshold
                            )

                            if matrix_1.shape[0] <= min_ion_num:
                                combination_list = matrix_1.index.values.tolist()
                            else:
                                combination_list = matrix_1.iloc[
                                    0:min_ion_num, :
                                ].index.values.tolist()
                            combination_result_df.loc[
                                str(targeted_compound), "Ion_Combination"
                            ] = combination_list
                            solo_list.append(targeted_compound)
                else:
                    col_name = list(temp_df.columns)

                    col_name = list(map(float, col_name))
                    new_com = [[x] for x in col_name]

                    difference_count_df_1 = (
                        calculate_average_score_and_difference_count(
                            targeted_compound,
                            new_com,
                            temp_df,
                            similarity_threshold,
                            fr_factor,
                        )
                    )

                    combination_df = difference_count_df_1[
                        difference_count_df_1["Diff_Count"]
                        >= difference_count_df_1.iat[0, 0]
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
                            ).sort_values(
                                by="com_score", inplace=False, ascending=False
                            )[
                                :5
                            ]

                    ion_list = list(temp_df)
                    combination_array = combination_df.index.values
                    n = 0
                    ion_num = 1

                    flag = True

                    while True:
                        if (
                            int((combination_df.max()).iloc[0])
                            >= int(temp_df.shape[0] - 1)
                            and ion_num >= min_ion_num
                        ):
                            break
                        elif not flag:  # Fixed equality comparison
                            break
                        else:
                            n = n + 1
                            total_list = []
                            new_total = []

                            for ion_combination in combination_array:

                                ion_combination_list = re.findall(
                                    r"\d+\.?\d*", ion_combination
                                )
                                ion_combination_list = list(
                                    map(float, ion_combination_list)
                                )
                                candidate_list = [
                                    i for i in ion_list if i not in ion_combination_list
                                ]

                                if candidate_list == []:
                                    if int((combination_df.max())[0]) >= int(
                                        temp_df.shape[0] - 1
                                    ):
                                        if combination_df.shape[0] > 1:
                                            combination_df = (
                                                filter_and_sort_combinations(
                                                    combination_df,
                                                    "Similar_Compound_Ave_Score",
                                                )
                                            )
                                            if combination_df.shape[0] > 1:
                                                combination_df = (
                                                    calculate_combination_score(
                                                        combination_df,
                                                        targeted_compound,
                                                        temp_df,
                                                        prefer_mz_threshold,
                                                    )
                                                )
                                                combination_df = (
                                                    combination_df.sort_values(
                                                        by="com_score",
                                                        inplace=False,
                                                        ascending=False,
                                                    )
                                                )
                                                combination_df = combination_df[:1]
                                            else:
                                                combination_df = combination_df
                                        else:
                                            combination_df = combination_df
                                        combination_array = combination_df.index.values
                                        combination_result_df.loc[
                                            str(targeted_compound),
                                            "Ion_Combination",
                                        ] = combination_array[0]
                                        combination_result_df.loc[
                                            str(targeted_compound), "Note"
                                        ] = "Despite the qualitative ion number is less than the defined number, its separation score reaches 1"
                                        flag = False

                                        break

                                    else:
                                        combination_result_df.loc[
                                            str(targeted_compound),
                                            "Ion_Combination",
                                        ] = "NA"
                                        combination_result_df.loc[
                                            str(targeted_compound), "Note"
                                        ] = "No ions available, this compound is discarded"
                                        flag = False
                                        break

                                elif flag:  # Fixed equality comparison

                                    for candidate in candidate_list:
                                        temp_ion_combination_list = (
                                            ion_combination_list.copy()
                                        )
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
                                        combination_result_df.loc[
                                            str(targeted_compound),
                                            "Ion_Combination",
                                        ] = "NA"
                                        combination_result_df.loc[
                                            str(targeted_compound), "Note"
                                        ] = (
                                            "Error: The 'difference_count_df' is "
                                            "empty."
                                        )
                                        flag = False

                                        break

                                    if combination_df.shape[0] > 1:
                                        combination_df = filter_and_sort_combinations(
                                            combination_df, "Similar_Compound_Ave_Score"
                                        )
                                        if combination_df.shape[0] > 1:
                                            combination_df = (
                                                calculate_combination_score(
                                                    combination_df,
                                                    targeted_compound,
                                                    temp_df,
                                                    prefer_mz_threshold,
                                                )
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
                            combination_result_df.loc[
                                str(targeted_compound), "Ion_Combination"
                            ] = combination_array[0]

    error_df = pd.DataFrame(columns=["Name", "Error"])
    name_list_total = []
    num = []
    name_list = combination_result_df.index.values.tolist()
    RT_list_total = []
    for name in name_list:
        if name in RT_data.index.values.tolist():
            if isinstance(combination_result_df.loc[name, "Ion_Combination"], str):  # Fixed type check
                ion_str = combination_result_df.loc[name, "Ion_Combination"]
                ion_list = re.findall(r"\d+\.?\d*", ion_str)

                ion_list = list(map(float, ion_list))
                for x in range(0, len(ion_list)):
                    name_list_total.append(name)
                    RT_list_total.append(RT_data.loc[name, "RT"])
                    num.append(ion_list[x])
            elif isinstance(combination_result_df.loc[name, "Ion_Combination"], list):  # Fixed type check
                ion_list = [
                    int(x) for x in combination_result_df.loc[name, "Ion_Combination"]
                ]
                for x in range(0, len(ion_list)):
                    name_list_total.append(name)
                    RT_list_total.append(RT_data.loc[name, "RT"])
                    num.append(ion_list[x])
            else:
                error_df.loc[len(error_df)] = [
                    name,
                    "The ion group format is incorrect.",
                ]
        else:
            error_df.loc[len(error_df)] = [
                name,
                "This compound is not in the RT list.",
            ]

    data = {"RT": RT_list_total, "ion": num}

    ion_rt = pd.DataFrame(data, index=name_list_total)

    ion_rt.sort_values(by="RT", inplace=True, ascending=True)
    for idx, row in ion_rt.iterrows():
        if row["RT"] > retention_time_max:
            ion_rt.loc[idx, "RT"] = retention_time_max

    write_msp(ion_rt, output_directory, msp_file_path)
