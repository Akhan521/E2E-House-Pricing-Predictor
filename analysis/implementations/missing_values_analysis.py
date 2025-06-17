from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''
This module uses the Template Method Design Pattern to define a template for missing values analysis.
The idea is to create a template that outlines the steps for analyzing missing values in a dataframe.
Subclasses must implement the specific methods for identifying and visualizing missing values.
'''

# Our abstract base class for missing values analysis (template):
# ----------------------------------------------------------------
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs analysis of missing values by identifying and visualizing them.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method identifies and visualizes missing values in the dataframe.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method should print the count of missing values for each column.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: This method should create a visualization (e.g., heatmap) of missing values.
        """
        pass


# Our concrete implementation of the missing values analysis template:
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each column in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the count of missing values for each column to the console.
        """
        print("\nCounts of Missing Values by Column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


# Missing Values Analysis Example:
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('../../extracted_data/AmesHousing.csv')

    # Perform Missing Values Analysis
    mv_analyzer = SimpleMissingValuesAnalysis()
    mv_analyzer.analyze(df)
