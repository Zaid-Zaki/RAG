from flask import Flask, request, jsonify, render_template, redirect, url_for

import torch
import torch.nn as nn


# Define the model structure (same as the trained model)
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = IrisClassifier()
model.load_state_dict(torch.load('iris_model.pth'))
model.eval()  # Set the model to evaluation mode


# Root route with a form for inputting data
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get data from form input
        features = [
            float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4'])
        ]

        # Convert data to tensor
        input_data = torch.tensor([features], dtype=torch.float32)

        # Forward pass to get predictions
        with torch.no_grad():
            output = model(input_data)
            _, predicted = torch.max(output, 1)

        # Redirect to the display page with the data and prediction
        return redirect(url_for('display',
                                feature1=features[0],
                                feature2=features[1],
                                feature3=features[2],
                                feature4=features[3],
                                prediction=predicted.item()))

    return render_template('index.html')


# Display data and prediction
@app.route('/display')
def display():
    feature1 = request.args.get('feature1')
    feature2 = request.args.get('feature2')
    feature3 = request.args.get('feature3')
    feature4 = request.args.get('feature4')
    prediction = request.args.get('prediction')

    return render_template('display.html',
                           feature1=feature1,
                           feature2=feature2,
                           feature3=feature3,
                           feature4=feature4,
                           prediction=prediction)


# Define a route for predictions via API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json  # Expects a JSON body

        # Convert data to tensor
        input_data = torch.tensor([data['features']], dtype=torch.float32)  # Use a batch format

        # Forward pass to get predictions
        with torch.no_grad():
            output = model(input_data)
            _, predicted = torch.max(output, 1)

        # Return prediction as JSON
        return jsonify({'prediction': predicted.item()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
