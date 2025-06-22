from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# This abstract base class defines the interface for bivariate analysis strategies.
# Subclasses must implement the analyze method to perform specific analyses.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the features to be analyzed.
        feature1 (str): The first feature/column name to be analyzed.
        feature2 (str): The second feature/column name to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Bivariate analysis strategy for numerical vs. numerical features
# -----------------------------------------------------------------
# This strategy analyzes the relationship between two numerical features using a scatter plot.
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the features to be analyzed.
        feature1 (str): The first numerical feature/column name to be analyzed.
        feature2 (str): The second numerical feature/column name to be analyzed.

        Returns:
        None: Displays a scatter plot showing the relationship between the two features.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Bivariate analysis strategy for categorical vs. numerical features
# -------------------------------------------------------------------
# This strategy analyzes the relationship between a categorical feature and a numerical feature using a box plot.
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the features to be analyzed.
        feature1 (str): The categorical feature/column name to be analyzed.
        feature2 (str): The numerical feature/column name to be analyzed.

        Returns:
        None: Displays a box plot showing the relationship between the two features.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


# Context class for bivariate analysis strategies
# -------------------------------------------------
# This class allows us to switch between different bivariate analysis strategies.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The bivariate analysis strategy to be used.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new bivariate analysis strategy to be used.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes bivariate analysis on two features using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the features to be analyzed.
        feature1 (str): The first feature/column name to be analyzed.
        feature2 (str): The second feature/column name to be analyzed.

        Returns:
        None: Executes the current strategy's analyze method to visualize the relationship between the two features.
        """
        self._strategy.analyze(df, feature1, feature2)


# Example usage of the BivariateAnalyzer with different strategies.
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('../../extracted_data/AmesHousing.csv')

    # Analyzing the relationship between two numerical features
    analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
    analyzer.execute_analysis(df, 'Gr Liv Area', 'SalePrice')

    # Analyzing the relationship between a categorical feature and a numerical feature
    analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    analyzer.execute_analysis(df, 'Overall Qual', 'SalePrice')
