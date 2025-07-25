import re

import numpy as np
import pandas as pd


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
    combination_df: pd.DataFrame,
    targeted_compound: str,
    temp_df: pd.DataFrame,
    prefer_mz_threshold: float,
) -> pd.DataFrame:
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
        ion_list = get_ion_list(index)
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


def get_ion_list(index: str) -> list[float]:
    ion_list = re.findall(r"\d+\.?\d*", index)
    ion_list = list(map(float, ion_list))
    return ion_list


def calculate_solo_compound_combination_score(
    matrix_1: pd.DataFrame, prefer_mz_threshold: float
) -> pd.DataFrame:
    """
    Calculate the combination score for solo compounds in a DataFrame.

    Args:
        matrix_1 (pd.DataFrame): DataFrame containing solo compound data.
        prefer_mz_threshold (int): The preferred m/z threshold.

    Returns:
        pd.DataFrame: DataFrame with combination scores added and sorted by score.

    """
    solo_scores = matrix_1.copy()
    solo_scores["ion"] = matrix_1["ion"].apply(
        lambda x: 1 if x < prefer_mz_threshold else x
    )
    solo_scores["com_score"] = solo_scores.apply(
        lambda row: pow(row.iloc[0], 0.5) * pow(row.iloc[1], 3), axis=1
    )
    return solo_scores.sort_values(by="com_score", ascending=False)
