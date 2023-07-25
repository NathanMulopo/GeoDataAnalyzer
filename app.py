from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

app = Flask(__name__)

# Function to generate visualizations
def generate_visualizations(df):
    # Visualization 1: Bar chart of Rock Types
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Rock Type', data=df)
    plt.title('Distribution of Rock Types')
    plt.xlabel('Rock Type')
    plt.ylabel('Count')
    bar_chart = mpld3.fig_to_html(plt.gcf())  # Convert the figure to HTML
    plt.close()  # Close the plot

    # Visualization 2: Scatter plot of Depth vs. Mineral Content
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Depth', y='Mineral Content', data=df)
    plt.title('Depth vs. Mineral Content')
    plt.xlabel('Depth')
    plt.ylabel('Mineral Content')
    scatter_plot = mpld3.fig_to_html(plt.gcf())  # Convert the figure to HTML
    plt.close()  # Close the plot

    # ... Add more visualizations as needed ...

    return bar_chart, scatter_plot

# Route for the home page
@app.route('/')
def index():
    # Load the geological data from the CSV file
    data_file = "data/geological_data.csv"
    df = pd.read_csv(data_file)

    # Generate visualizations
    bar_chart, scatter_plot = generate_visualizations(df)

    # Calculate average temperature
    average_temperature = df['Temperature'].mean()

    # Calculate max depth per location
    max_depth_per_location = df.groupby('Location')['Depth'].max().reset_index()

    # Calculate mineral content count
    mineral_content_count = df['Mineral Content'].value_counts()

    # ... Code for other visualizations ...

    return render_template('index.html', bar_chart=bar_chart, scatter_plot=scatter_plot,
                           average_temperature=average_temperature,
                           max_depth_per_location=max_depth_per_location,
                           mineral_content_count=mineral_content_count)

# Route for the machine learning prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Load the geological data from the CSV file
    data_file = "data/geological_data.csv"
    df = pd.read_csv(data_file)

    # Ensure 'Rock Type', 'Mineral Content', and 'Location' columns are string data types
    df['Rock Type'] = df['Rock Type'].astype(str)
    df['Mineral Content'] = df['Mineral Content'].astype(str)
    df['Location'] = df['Location'].astype(str)

    # One-hot encode the 'Rock Type', 'Mineral Content', and 'Location' columns
    label_encoder = LabelEncoder()
    df['Rock Type'] = label_encoder.fit_transform(df['Rock Type'])
    df['Mineral Content'] = label_encoder.fit_transform(df['Mineral Content'])
    df['Location'] = label_encoder.fit_transform(df['Location'])

    # Prepare the data for the machine learning model
    X = df.drop(columns=['Rock Type'])  # Features are all columns except 'Rock Type'
    y = df['Rock Type']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the decision tree classifier
    clf = DecisionTreeClassifier()

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Decode the predicted numeric labels back to original 'Rock Type' labels
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # ... Code for other visualizations ...

    return render_template('predict.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
