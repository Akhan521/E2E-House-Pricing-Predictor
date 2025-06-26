import logging
from abc import ABC, abstractmethod
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# An abstract class that will define our missing value handling interface.
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing missing values to be handled.

        Returns:
        pd.DataFrame: The cleaned DataFrame with missing values handled.
        """
        pass


# Our first concrete strategy: Drop Missing Values
# -----------------------------------------------------
# This strategy drops rows or columns with missing values based on the specified axis and threshold.
# If the number of non-NA values in a row/column is less than the threshold, the given row/column will be dropped.
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, threshold=None):
        """
        Initializes the DropMissingValuesStrategy with the specified axis and threshold.

        Parameters:
        axis (int): 0 or 1 -> 0 for dropping rows w/ missing values, 1 for dropping columns w/ missing values.
        threshold (int): The threshold for non-NA values. Rows/Columns with less than threshold non-NA values are dropped.
        """
        self.axis = axis
        self.threshold = threshold

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows/columns w/ missing values based on the provided axis and threshold.

        Parameters:
        df (pd.DataFrame): The DataFrame containing missing values to be handled.

        Returns:
        pd.DataFrame: The cleaned DataFrame with missing values dropped.
        """
        logging.info(f"Dropping missing values with axis={self.axis} and threshold={self.threshold}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.threshold)
        logging.info("Missing values dropped.")
        return df_cleaned


# Our second concrete strategy: Fill Missing Values
# -----------------------------------------------------
# This strategy fills missing values using a specified method (mean, median, mode, or constant).
# If the method is 'constant', a specific fill value can be provided.
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method for filling missing values. Options are ('mean', 'median', 'mode', 'constant').
        fill_value (any): If the method is 'constant', a specific fill value can be provided.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method or constant value.

        Parameters:
        df (pd.DataFrame): The DataFrame containing missing values to be handled.

        Returns:
        pd.DataFrame: The cleaned DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            # Only numerical columns are to be filled with their mean.
            numerical_cols = df_cleaned.select_dtypes(include="number").columns # The names of our numerical columns
            df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(
                df[numerical_cols].mean()
            )
        elif self.method == "median":
            # Only numerical columns are to be filled with their median.
            numerical_cols = df_cleaned.select_dtypes(include="number").columns # The names of our numerical columns
            df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(
                df[numerical_cols].median()
            )
        elif self.method == "mode":
            # For both numerical and categorical columns, we fill with the mode.
            for col in df_cleaned.columns:
                df_cleaned[col].fillna(df[col].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            # Fill all columns with the constant value.
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"'{self.method}' is an unknown method... No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned


# Our conext class that allows us to switch between different missing value handling strategies.
# This class uses the Strategy Design Pattern to allow for flexible missing value handling.
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The missing value handling strategy to be used.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new missing value handling strategy to be used.
        """
        logging.info("Switching to a new missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the current missing value handling strategy on the provided DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing missing values to be handled.

        Returns:
        pd.DataFrame: The cleaned DataFrame with missing values handled according to the current strategy.
        """
        logging.info("Executing the current missing value handling strategy.")
        return self._strategy.handle(df)


# Example usage of the MissingValueHandler with different strategies.
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('../../extracted_data/AmesHousing.csv')

    # Init. the missing value handler with a specific strategy
    missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, threshold=5))
    df_cleaned = missing_value_handler.handle_missing_values(df)

    # Switch to a different strategy: Fill Missing Values with Mean
    missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    df_filled = missing_value_handler.handle_missing_values(df)
