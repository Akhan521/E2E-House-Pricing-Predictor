import pandas as pd
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)
from zenml import step


@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Handles missing values using the MissingValueHandler with the specified strategy."""
    # We drop rows/columns with missing values or fill them based on the strategy provided.
    if strategy == "drop":
        mv_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0)) # Default to dropping rows with missing values.
    elif strategy in ["mean", "median", "mode", "constant"]:
        mv_handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"The provided missing value handling strategy is unsupported/unknown: {strategy}")

    df_cleaned = mv_handler.handle_missing_values(df)
    return df_cleaned
