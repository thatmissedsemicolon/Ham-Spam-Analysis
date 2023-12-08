"""
Flask application for classifying emails as ham or spam using a logistic regression model.
Allows users to upload a CSV or .data file, analyzes it, and displays the results.
"""

import base64
from io import BytesIO
from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Set the backend to Agg before other imports

app = Flask(__name__)

# Define the logistic regression model
model = LogisticRegression(solver='liblinear')

def allowed_file(filename):
    """
    Checks if the file is a .data or .csv file.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'data', 'csv'}

def validate_file_content(file_stream, file_extension):
    """
    Validate the content of the file based on its extension.
    Checks for the correct format and that the file is not empty.
    """
    try:
        # Read the file into a DataFrame
        df = pd.read_csv(file_stream, header=None if file_extension == 'data' else 0)

        # Reset the file stream position to the beginning
        file_stream.seek(0)

        # Check if the DataFrame is empty
        if df.empty:
            return False, None

        # Specific checks for .data files from Spambase
        if file_extension == 'data' and df.shape[1] != 58:
            return False, None

        return True, df
    except Exception:
        # Handle exceptions for invalid file formats
        return False, None

def plot_to_base64(plt_figure):
    """
    Converts a matplotlib plot to a base64 encoded string.
    """
    img = BytesIO()
    plt_figure.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def process_file(file_stream, filename):
    """
    Processes the uploaded file based on its format.
    """
    if filename.endswith('.data'):  # spambase.data format
        column_names = [f'feature_{i}' for i in range(57)] + ['label']
        df = pd.read_csv(file_stream, header=None, names=column_names)
    else:  # CSV format
        df = pd.read_csv(file_stream)
    return df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Route for uploading and processing the file.
    """
    if request.method == 'POST':
        file = request.files.get('file')
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''

        if file and allowed_file(file.filename):
            is_valid, df = validate_file_content(file.stream, file_extension)
            if is_valid:
                df = process_file(file.stream, file.filename)
                features = df.drop('label', axis=1)
                labels = df['label']
                features_train, features_test, labels_train, labels_test = \
                    train_test_split(features, labels, test_size=0.2, random_state=42)
                model.fit(features_train, labels_train)
                predictions = model.predict(features_test)
                accuracy = accuracy_score(labels_test, predictions)

                # Plotting
                plt.figure()
                categories = ['Ham', 'Spam']
                counts = [(predictions == 0).sum(), (predictions == 1).sum()]
                plt.bar(categories, counts, color=['blue', 'red'])
                plt.xlabel('Category')
                plt.ylabel('Count')
                plt.title('Ham vs Spam Analysis')
                plot_url = plot_to_base64(plt)

                return render_template(
                    'result.html', 
                    accuracy=round(accuracy, 4),
                    plot_url=plot_url
                )
            return render_template(
                'upload.html', 
                error='Invalid file content. Please upload a valid file!'
            )
        return render_template(
            'upload.html', 
            error='Invalid file type. Please upload a valid .data or .csv file!'
        )
    return render_template('upload.html', error=None)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8000)
