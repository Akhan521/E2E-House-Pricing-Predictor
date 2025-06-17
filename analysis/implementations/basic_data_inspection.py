from abc import ABC, abstractmethod
import pandas as pd

'''
This module uses the Strategy Design Pattern to implement different data inspection strategies.
The core idea is to define an interface for data inspection strategies and provide concrete implementations for specific inspection tasks.
We can use these strategies within some context to perform various inspections on our data.
'''

# Our abstract base class for data inspection strategies (common interface):
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform some form of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: This method should print the results directly.
        """
        pass


# Our first strategy: Data Types Inspection
# -----------------------------------------------------
# This strategy inspects the data types and non-null counts of the dataframe columns.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints information about the data types and non-null counts of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints information about data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())


# Our second strategy: Summary Statistics Inspection
# -----------------------------------------------------
# This strategy provides summary statistics for numerical and categorical features in the dataframe.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for our data's numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


# Our context class that allows us to switch between different inspection strategies.
# This class uses the Strategy Design Pattern to allow for flexible data inspection.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Sets up our DataInspector with a specific data inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The data inspection strategy to be used.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for data inspection.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes data inspection on our data using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspect method.
        """
        self._strategy.inspect(df)

# DataInspector Example W/ Different Strategies:
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('../../extracted_data/AmesHousing.csv')

    # Init. the Data Inspector with a specific strategy
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.execute_inspection(df)

    # Change your strategy to use a different inspection method
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.execute_inspection(df)
