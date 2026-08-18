import os
from ast import literal_eval

import pandas as pd

# Point these at real files (env vars or edit the defaults) before running.
INPUT_CSV = os.environ.get("PROCESSED_RESULTS_CSV", "data/processed_results.csv")
OUTPUT_CSV = os.environ.get("COMPUTED_AVERAGES_CSV", "data/computed_averages.csv")

# (source column in the processed-results CSV, inner dict key nested inside
# each cell, summary column label, name for the per-row mean column).
# These strings must match main.py's SUBSCALES table exactly — several use
# a different casing for the source column vs. the inner key on purpose,
# carried over unchanged from the original data.
SUBSCALES = [
    ("interest enjoyment", "Interest_enjoyment", "Interest Enjoyment", "mean_interest_enjoyment"),
    ("perceived competence", "Perceived competence", "Perceived Competence", "mean_perceived_competence"),
    ("Effort importance", "Effort importance", "Effort Importance", "mean_effort_importance"),
    ("Pressure Tension", "Pressure tension", "Pressure Tension", "mean_pressure_tension"),
    ("Perceived choice", "Perceived choice", "Perceived Choice", "mean_perceived_choice"),
    ("Value Usefulness", "Value Usefulness", "Value Usefulness", "mean_value_usefulness"),
]


def extract_subscale(raw_cell, inner_key):
    """Parse one cell of the processed-results CSV (a stringified list of
    dicts) and pull out the answers dict for `inner_key`. Returns None if
    the cell can't be parsed or the key isn't present."""
    try:
        nested = literal_eval(raw_cell)
        return nested[0].get(inner_key)
    except Exception as e:
        print(f"Error processing row ({inner_key}): {e}")
        return None


def subscale_mean_frame(data, source_column, inner_key, mean_column):
    """Build the per-question answers frame for one subscale and add its
    row-wise mean. Returns (frame, mean_column) or (empty_frame, None) if
    there was nothing valid to parse."""
    parsed = data[source_column].apply(lambda cell: extract_subscale(cell, inner_key))
    # A row that failed to parse yields None rather than a dict; pandas
    # can't build a DataFrame from a list mixing dicts and None (it raises
    # TypeError), so failed rows become an empty dict instead — they end up
    # as an all-NaN row, which is what "no valid data for this row" should
    # look like, instead of taking down the whole script.
    safe_parsed = parsed.apply(lambda v: v if isinstance(v, dict) else {})
    frame = pd.DataFrame(safe_parsed.tolist(), index=data.index)
    if frame.empty or frame.isna().all(axis=None):
        print(f"No valid data to display for {source_column}.")
        return frame, None
    frame[mean_column] = frame.mean(axis=1).round(3)
    print(frame.head())
    return frame, mean_column


def build_summary(data):
    summary = pd.DataFrame(index=data.index)
    for source_column, inner_key, label, mean_column in SUBSCALES:
        frame, computed_col = subscale_mean_frame(data, source_column, inner_key, mean_column)
        summary[label] = frame[computed_col] if computed_col else pd.NA
    return summary


if __name__ == "__main__":
    data = pd.read_csv(INPUT_CSV)
    print("Column names:", data.columns.tolist())

    summary_data = build_summary(data)
    summary_data.to_csv(OUTPUT_CSV)
    print(summary_data.head())
