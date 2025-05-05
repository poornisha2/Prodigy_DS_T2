# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset from seaborn
df = sns.load_dataset('iris')

# Display basic info
print("Shape of the dataset:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Basic stats
print("\nSummary statistics:")
print(df.describe())

# Check class distribution
print("\nSpecies distribution:")
print(df['species'].value_counts())

# Pairplot to visualize relationships
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Boxplot for each feature by species
for column in df.columns[:-1]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='species', y=column, data=df)
    plt.title(f"{column.capitalize()} Distribution by Species")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
