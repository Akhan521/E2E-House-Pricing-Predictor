from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# This abstract base class defines the interface for univariate analysis strategies.
# Subclasses must implement the analyze method to perform specific analyses.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature to be analyzed.
        feature (str): The feature/column name to be analyzed.

        Returns:
        None: This method visualizes the feature's distribution.
        """
        pass


# Univariate analysis for numerical features
# --------------------------------------------------
# This strategy analyzes numerical features by plotting their distribution using a histogram and KDE.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE (Kernel Density Estimate).

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature to be analyzed.
        feature (str): The numerical feature/column name to be analyzed.

        Returns:
        None: Displays a histogram with a KDE plot.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# Univariate analysis for categorical features
# --------------------------------------------------
# This strategy analyzes categorical features by plotting their distribution using a bar plot.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature to be analyzed.
        feature (str): The categorical feature/column name to be analyzed.

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# Context class for univariate analysis strategies
# --------------------------------------------------
# This class allows us to switch between different univariate analysis strategies.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The univariate analysis strategy to be used.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new univariate analysis strategy to be used.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes univariate analysis on a specific feature using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the feature to be analyzed.
        feature (str): The feature/column name to be analyzed.

        Returns:
        None: Executes the current strategy's analyze method to visualize the feature's distribution.
        """
        self._strategy.analyze(df, feature)


# Example usage of the UnivariateAnalyzer with different strategies.
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('../../extracted_data/AmesHousing.csv')

    # Analyzing a numerical feature (e.g., SalePrice)
    analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    analyzer.execute_analysis(df, 'SalePrice')

    # Analyzing a categorical feature (e.g., Neighborhood)
    analyzer.set_strategy(CategoricalUnivariateAnalysis())
    analyzer.execute_analysis(df, 'Neighborhood')
