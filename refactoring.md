# Refactoring Plan for wtv Package

This document outlines potential refactoring improvements for the wtv package, leveraging the matchms library where appropriate.

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [matchms Integration Opportunities](#matchms-integration-opportunities)
3. [Similarity Calculation Improvements](#similarity-calculation-improvements)
4. [Code Structure Refactoring](#code-structure-refactoring)
5. [Type Safety Improvements](#type-safety-improvements)
6. [Retention Time/Index Flexibility](#retention-timeindex-flexibility)
7. [Implementation Priority](#implementation-priority)

---

## Current Architecture Analysis

### Key Components

#### 1. Similarity Module ([wtv/similarity.py](wtv/similarity.py))

The current implementation uses custom distance/similarity functions:

| Function | Purpose | Issues |
|----------|---------|--------|
| `dot_product_distance()` | Calculates dot product similarity between two vectors | Custom implementation, not standardized |
| `weighted_dot_product_distance()` | Weighted version with FR (fragment ratio) calculation | Complex logic, hard to maintain |
| `calculate_similarity()` | Computes similarity between target and all compounds in DataFrame | Iterative, not vectorized |
| `calculate_average_score_and_difference_count()` | Averages scores across ion combinations | Nested dependencies |
| `calculate_combination_score()` | Scores ion combinations based on m/z preference | Magic numbers (e.g., exponent values) |
| `calculate_solo_compound_combination_score()` | Single compound scoring | Simple but could be unified |

**Key Observations:**
- Uses custom dot-product based similarity instead of established mass spectrometry metrics
- Fragment Ratio (FR) calculation is unique to this implementation
- No use of existing mass spectrometry libraries for core calculations

#### 2. Ion Selection Module ([wtv/ion_selection.py](wtv/ion_selection.py))

Core workflow:
1. Load MSP data → create ion matrix and RT data
2. Find nearby compounds within RT window
3. Filter ions by intensity threshold
4. Calculate similarity to find similar compounds
5. Generate optimal ion combinations
6. Write filtered spectra to MSP

**Issues Identified:**
- `ThreadPoolExecutor` is commented out (line 334-335) - parallelization disabled
- Complex while loops in `calculate_ion_combination()` (lines 326-437) - hard to follow
- Type casting issues: `int()` conversions that may lose precision (line 161, 92)
- Mixed string/list handling for `Ion_Combination` field (lines 77-97)

#### 3. Utilities Module ([wtv/utils.py](wtv/utils.py))

Already uses matchms for MSP I/O:
- `read_msp()` - Uses `load_from_msp()` from matchms
- `write_msp()` - Uses `save_as_msp()` from matchms
- `normalize_array()` - Custom normalization (could use matchms filtering)

**Issues:**
- Line 117: `if ions is not "NA"` - should use `!=` not `is not` for string comparison
- Custom normalization could leverage matchms' built-in filters

---

## matchms Integration Opportunities

### Available Similarity Metrics in matchms

Based on the matchms library documentation, the following similarity metrics are available:

| Metric Class | Description | Use Case for wtv |
|--------------|-------------|------------------|
| `CosineGreedy` | Standard cosine similarity with greedy peak matching | Could replace custom dot product |
| `ModifiedCosine` | Cosine with global mass shift optimization | Better for cross-instrument comparisons |
| `NearestCosine` | Finds nearest neighbor peaks before scoring | Alternative peak alignment |
| `IntersectMz` | Simple m/z intersection count | Quick filtering |
| `FingerprintSimilarity` | Jaccard/Tanimoto on binary fingerprints | Fast approximate similarity |
| `Spec2Vec` | Machine learning-based spectrum similarity | Advanced alternative (requires separate install) |
| `MS2DeepScore` | Deep learning similarity score | State-of-the-art (requires separate install) |

### Filtering Functions Available in matchms

| Function | Purpose | wtv Equivalent |
|----------|---------|----------------|
| `select_by_mz(mz_min, mz_max)` | Filter peaks by m/z range | `create_ion_matrix()` filtering |
| `select_by_intensity(min, max)` | Filter by absolute intensity | `filter_matrix()` |
| `select_by_relative_intensity(min%, max%)` | Filter by % of base peak | `min_ion_intensity_percent` logic |
| `remove_peaks_around_precursor_mz()` | Remove precursor region peaks | Not currently needed |
| `remove_peaks_uncharged()` | Remove uncharged peaks | Data cleaning |
| `normalize_intensities()` | Scale intensities to 0-100 | `normalize_array()` |

### Pipeline Class for Workflow Definition

matchms provides a `Pipeline` class for defining reproducible workflows via YAML:

```yaml
pipeline:
  - ImportFromMSP
  - SelectByMz: {mz_min: 35, mz_max: 400}
  - NormalizeIntensities
  - CalculateSimilarity: {metric: "CosineGreedy", tolerance: 0.1}
  - ExportToMSP
```

**Benefit for wtv:** Could define the entire ion selection workflow as a YAML configuration file.

### Recommended matchms Integrations

#### High Priority

1. **Replace custom normalization** with `matchms.filtering.normalize_intensities()`
   - Current: `normalize_array()` in utils.py
   - Benefit: Standardized, tested implementation

2. **Use matchms filtering pipeline** for peak selection
   - Current: Manual filtering in `filter_matrix()` and `create_ion_matrix()`
   - Benefit: Consistent API, composable operations

3. **Adopt matchms `SpectrumProcessor`** for batch processing
   - Current: Custom iteration over spectra
   - Benefit: Built-in logging, progress tracking, caching

#### Medium Priority

4. **Evaluate `CosineGreedy` or `ModifiedCosine`** as alternatives to custom dot product
   - Current: `weighted_dot_product_distance()` with custom weighting
   - Consideration: May need to preserve FR calculation, so full replacement may not be possible

5. **Use `calculate_scores()`** for matrix computations
   - Current: Manual nested loops in `calculate_similarity()`
   - Benefit: Optimized pairwise calculations, sparse result storage

---

## Similarity Calculation Improvements

### Current Algorithm Analysis

The current `weighted_dot_product_distance()` function:

```python
def weighted_dot_product_distance(compare_df: pd.DataFrame, fr_factor: float) -> float:
    m_q = pd.Series(compare_df.index)  # m/z values
    m_q = m_q.astype(float)
    i_q = np.array(compare_df.iloc[:, 0])  # intensities spec 1
    i_r = np.array(compare_df.iloc[:, 1])  # intensities spec 2
    k = 0.5
    exponent = 2
    w_q = np.power(i_q, k) * np.power(m_q, exponent)  # weighted
    w_r = np.power(i_r, k) * np.power(m_q, exponent)
    ss = dot_product_distance(w_q, w_r)
    # ... FR calculation ...
```

**Strengths:**
- Weights by both intensity and m/z
- Includes fragment ratio (FR) calculation for peak pattern similarity

**Weaknesses:**
- Magic numbers (k=0.5, exponent=2) not documented
- Not comparable to standard similarity scores
- Hard to tune for different use cases

### Proposed Improvements

#### Option A: Hybrid Approach (Recommended)

Keep the custom weighting but make it configurable:

```python
def weighted_dot_product_distance(
    compare_df: pd.DataFrame,
    fr_factor: float,
    intensity_weight: float = 0.5,
    mz_weight: float = 2.0,
) -> float:
    """Calculate weighted dot product with configurable weights.
    
    Args:
        intensity_weight: Exponent for intensity weighting (default 0.5)
        mz_weight: Exponent for m/z weighting (default 2.0)
    """
    m_q = pd.Series(compare_df.index).astype(float)
    i_q = np.array(compare_df.iloc[:, 0])
    i_r = np.array(compare_df.iloc[:, 1])
    
    w_q = np.power(i_q, intensity_weight) * np.power(m_q, mz_weight)
    w_r = np.power(i_r, intensity_weight) * np.power(m_q, mz_weight)
    # ... rest unchanged
```

#### Option B: Standard Cosine with Custom Post-Processing

Use matchms' optimized cosine calculation, then apply FR adjustment:

```python
from matchms.similarity import CosineGreedy

def hybrid_similarity(target_spectrum, reference_spectrum, fr_factor: float):
    """Combine standard cosine with custom FR scoring."""
    cosine = CosineGreedy(tolerance=0.1)
    base_score = cosine.pair(target_spectrum, reference_spectrum)
    
    # Apply FR adjustment as multiplier
    fr_score = calculate_fragment_ratio(target_spectrum, reference_spectrum)
    return base_score * (1 + 0.1 * fr_score)
```

---

## Code Structure Refactoring

### Suggested Module Reorganization

```
wtv/
├── __init__.py
├── cli.py                    # CLI interface (unchanged)
├── core/                     # New: Core algorithms
│   ├── __init__.py
│   ├── ion_selection.py      # Main selection logic
│   └── combination_scoring.py # Scoring algorithms
├── similarity/               # New: Similarity module
│   ├── __init__.py
│   ├── base.py               # Base classes for similarity
│   ├── custom.py             # Custom wtv algorithms
│   └── matchms_adapters.py   # matchms compatibility layer
├── filtering/                # New: Peak filtering
│   ├── __init__.py
│   └── processors.py         # matchms-compatible processors
├── io/                       # New: I/O operations
│   ├── __init__.py
│   ├── msp_reader.py         # Wrap matchms loading
│   └── msp_writer.py         # Wrap matchms saving
└── utils/                    # Existing utilities
    ├── __init__.py
    ├── normalization.py      # Moved from root
    └── validation.py         # New: Input validation
```

### Key Refactoring Tasks

#### 1. Extract Ion Selection Logic

**Current:** All logic in `ion_selection.py` (~470 lines)

**After:** Split into focused modules:

```python
# core/ion_selection.py
def run_ion_selection(...):
    RT_data, matrix = load_data(...)
    combinations = generate_ion_combinations(...)
    ion_rt = collect_ion_retention_times(...)
    write_results(...)

# core/combination_scoring.py  
def score_ion_combination(ions, compound_data, config):
    # Isolated scoring logic
```

#### 2. Add Configuration Class

Create a structured configuration object:

```python
@dataclass
class IonSelectionConfig:
    mz_min: float = 0
    mz_max: float = np.inf
    rt_window: float = 1.0
    min_ion_intensity_percent: float = 5.0
    min_ion_num: int = 3
    prefer_mz_threshold: float = 150
    similarity_threshold: float = 0.85
    fr_factor: float = 2.0
    retention_time_max: float = np.inf
```

Benefits:
- Single source of truth for parameters
- Easier testing with default values
- JSON/YAML serialization support

#### 3. Enable Parallel Processing

Re-enable and properly implement ThreadPoolExecutor:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_ion_combinations(..., n_workers=None):
    ...
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_compound, args): args
            for args in compound_args_list
        }
        for future in as_completed(futures):
            results.append(future.result())
```

---

## Retention Time/Index Flexibility

The current implementation uses retention time from the `retention_time` metadata field. To support both retention time and retention index:

### Implementation Approach

1. **Add configurable retention field selection**
   - Add `--retention_field` CLI flag (default: "retention_time")
   - Can be set to "retention_index" when RI data is available
   - No external library dependency needed - just flexible field access

2. **Update data loading functions**
   - Pass retention field name through the pipeline
   - Update `get_rt_data()` to use configurable field

```python
def get_rt_data(retention_field: str, spectra):
    spectra_md, _ = get_metadata_as_array(spectra)
    df = (
        pd.DataFrame(spectra_md)
        .rename(columns={"compound_name": "Name", retention_field: "RT"})
        .get(["Name", "RT"])
    )
    df.set_index("Name", inplace=True)
    return df
```

This approach provides flexibility without the overhead of integrating RIAssigner. Users who need retention index calculation can preprocess their data with external tools and include the `retention_index` field in their MSP files.

---

## Type Safety Improvements

### Current Issues

1. **m/z values converted to int unintentionally:**
   ```python
   # ion_selection.py line 161
   new_temp_df["ion"] = new_temp_df["ion"].astype("int")  # Loses precision!
   ```

2. **Mixed types for Ion_Combination:**
   - Sometimes list: `list([300.0, 204.09])`
   - Sometimes string: `"[300.0, 350.0]"`

3. **Missing type hints** on many functions

### Proposed Fixes

#### 1. Preserve Float Types Throughout

```python
from typing import Union

def get_ion_list(index: str) -> list[float]:
    """Parse ion m/z values, preserving float precision."""
    ion_list = re.findall(r"\d+\.?\d*", index)
    return [float(x) for x in ion_list]  # Always float, never int

# For column operations
matrix = matrix.astype({col: float for col in matrix.columns})
```

#### 2. Standardize Ion_Combination Type

Always use list[float]:

```python
# In combination results
combination_result_df["Ion_Combination"] = combination_result_df["Ion_Combination"].apply(
    lambda x: x if isinstance(x, list) else parse_ion_string(x)
)

def parse_ion_string(ion_str: Union[str, list]) -> list[float]:
    if isinstance(ion_str, list):
        return [float(x) for x in ion_str]
    return [float(x) for x in re.findall(r"\d+\.?\d*", ion_str)]
```

#### 3. Add Comprehensive Type Hints

```python
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

def calculate_similarity(
    target_name: str, 
    df: pd.DataFrame, 
    fr_factor: float
) -> pd.DataFrame:
    ...

def filter_matrix(
    matrix: pd.DataFrame, 
    compound: str, 
    min_ion_intensity: float
) -> pd.DataFrame:
    ...
```

---

## Implementation Priority

### Phase 1: Quick Wins (Low Effort, High Impact)

| Task | Effort | Impact | Notes |
|------|--------|--------|-------|
| Fix `is not "NA"` to `!= "NA"` | 5 min | High | Bug fix in utils.py |
| Add type hints to public APIs | 2 hrs | Medium | Better IDE support |
| Create IonSelectionConfig dataclass | 1 hr | Medium | Cleaner parameter passing |
| Replace int() casts with float() | 30 min | High | Prevents data loss |

### Phase 2: matchms Integration (Medium Effort, High Impact)

| Task | Effort | Impact | Notes |
|------|--------|--------|-------|
| Replace normalize_array() with matchms | 1 hr | Medium | Standardized normalization |
| Use matchms filtering pipeline | 4 hrs | High | More maintainable |
| Adopt SpectrumProcessor for batch ops | 4 hrs | High | Better performance |
| Use calculate_scores() for matrices | 2 hrs | Medium | Optimized calculations |

### Phase 3: Structural Refactoring (High Effort, Medium Impact)

| Task | Effort | Impact | Notes |
|------|--------|--------|-------|
| Reorganize module structure | 8 hrs | Medium | Better maintainability |
| Re-enable parallel processing | 4 hrs | High | Performance improvement |
| Add comprehensive validation | 4 hrs | Medium | Better error messages |

### Phase 4: Advanced Features (Variable Effort, Variable Impact)

| Task | Effort | Impact | Notes |
|------|--------|--------|-------|
| Implement hybrid similarity (cosine + FR) | 8 hrs | Medium | Better accuracy |
| Add YAML workflow configuration | 4 hrs | Medium | User-friendly config |
| Add retention index field support | 2 hrs | Low | Simple field name parameter |

---

## Migration Strategy

### Backward Compatibility

All changes should maintain backward compatibility:

1. **Preserve function signatures** where possible
2. **Add deprecation warnings** for changed APIs
3. **Maintain output format** for MSP files
4. **Keep CLI interface** unchanged

### Testing Requirements

Before each phase:
- Run existing test suite
- Add tests for new functionality
- Verify output matches expected results

### Rollback Plan

1. Keep original code in git history
2. Feature flags for major changes
3. Gradual rollout with canary testing

---

## References

- [matchms Documentation](https://matchms.readthedocs.io/)
- [matchms GitHub Repository](https://github.com/matchms/matchms)
- [Original WTV_2.0 Paper](https://doi.org/10.1016/j.molp.2024.04.012)

---

*Last updated: 2026-07-09*
