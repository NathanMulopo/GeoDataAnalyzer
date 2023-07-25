import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the geological data from the CSV file
data_file = "data/geological_data.csv"
df = pd.read_csv(data_file)

# Display the first few rows of the DataFrame to get an overview of the data
print(df.head())

# Get basic statistics of the numerical columns
print(df.describe())

# Check the data types of columns and look for any missing values
print(df.info())

# Create a bar chart to visualize the count of each Rock Type
plt.figure(figsize=(8, 6))
sns.countplot(x='Rock Type', data=df)
plt.title('Count of Each Rock Type')
plt.xlabel('Rock Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Create a scatter plot to visualize the relationship between Depth and Temperature
plt.figure(figsize=(8, 6))
plt.scatter(df['Depth'], df['Temperature'])
plt.title('Depth vs. Temperature')
plt.xlabel('Depth')
plt.ylabel('Temperature')
plt.show()

# Calculate the average temperature for each rock type
average_temperature = df.groupby('Rock Type')['Temperature'].mean()
print("\nAverage Temperature for Each Rock Type:")
print(average_temperature)

# Calculate the maximum depth for each location
max_depth_per_location = df.groupby('Location')['Depth'].max()
print("\nMaximum Depth for Each Location:")
print(max_depth_per_location)

# Calculate the total count of each mineral content
mineral_content_count = df['Mineral Content'].value_counts()
print("\nTotal Count of Each Mineral Content:")
print(mineral_content_count)

# Conclusion and Summary

print("\n--- Conclusion and Summary ---\n")

# Summarize the project's objectives
print("Project Objectives:")
print("The goal of the GeoDataAnalyzer project was to explore and analyze sample geological data to gain insights into rock types, mineral content, locations, depths, and temperatures. We aimed to visualize the data and provide basic data analysis.")

# Provide key findings and insights
print("\nKey Findings and Insights:")
print("1. The bar chart displayed the frequency of each rock type in the dataset, providing an overview of rock type distribution.")
print("2. The scatter plot showed the relationship between depth and temperature, indicating possible correlations.")
print("3. Data analysis revealed the average temperature for each rock type, the maximum depth for each location, and the total count of each mineral content.")

# Project Outcomes
print("\nProject Outcomes:")
print("The GeoDataAnalyzer successfully analyzed the sample geological data, providing initial insights into the dataset. We visualized the data, calculated average temperatures, maximum depths, and mineral content counts. This project serves as a foundation for further data exploration and analysis in geological research and resource extraction.")

print("\n--- End of Project ---")
