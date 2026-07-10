# API Reference: Similarity

This document describes the similarity calculation functions in `wtv.similarity`.

## `dot_product_distance`

Calculate the dot product distance between two vectors.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `p` | np.ndarray | First vector |
| `q` | np.ndarray | Second vector |

### Returns

- `float`: Dot product distance between the two vectors (0 to 1)

### Example

```python
import numpy as np
from wtv.similarity import dot_product_distance

p = np.array([1, 2, 3])
q = np.array([4, 5, 6])
score = dot_product_distance(p, q)
print(f"Similarity score: {score}")
```

## `weighted_dot_product_distance`

Calculate the weighted dot product distance with fragment ratio consideration.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `compare_df` | pd.DataFrame | DataFrame with two columns representing vectors to compare |
| `fr_factor` | float | Factor used in the FR calculation |

### Returns

- `float`: Composite score based on weighted dot product distance and fragment ratio

## `calculate_similarity`

Calculate similarity scores between a target compound and all compounds in a DataFrame.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_name` | str | Name of the target compound |
| `df` | pd.DataFrame | DataFrame containing compound data |
| `fr_factor` | float | Factor used in the weighted dot product distance calculation |

### Returns

- `pd.DataFrame`: DataFrame containing similarity scores for each compound

## `calculate_average_score_and_difference_count`

Calculate the average similarity score and difference count for a targeted compound.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `targeted_compound` | str | Name of the targeted compound |
| `ion_combination` | list | List of ion combinations for similarity calculation |
| `df` | pd.DataFrame | DataFrame containing compound data |
| `similarity_threshold` | float | Similarity threshold for considering compounds as similar |
| `fr_factor` | float | Factor used in the weighted dot product distance calculation |

### Returns

- `pd.DataFrame`: DataFrame containing difference counts and average similarity scores

## `calculate_combination_score`

Calculate the combination score for ion combinations.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `combination_df` | pd.DataFrame | DataFrame containing ion combinations |
| `targeted_compound` | str | Name of the targeted compound |
| `temp_df` | pd.DataFrame | Temporary DataFrame containing compound data |
| `prefer_mz_threshold` | float | Preferred m/z threshold |

### Returns

- `pd.DataFrame`: DataFrame with combination scores added

## `calculate_solo_compound_combination_score`

Calculate the combination score for solo compounds.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix_1` | pd.DataFrame | DataFrame containing solo compound data |
| `prefer_mz_threshold` | float | Preferred m/z threshold |

### Returns

- `pd.DataFrame`: DataFrame with combination scores added and sorted by score
