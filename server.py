from flask import Flask, request, jsonify
import pandas as pd
from import_bert import import_bert
from predict_with_model import predict_with_model
from waitress import serve
import os

def create_app():
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        # 1. Get inputs from client requests
        data = request.json
        text = data['text']

        # 2. Embed the text from the input with import_bert()
        embedded_text = import_bert(pd.DataFrame({'text': [text]}), 'text')

        # 3. Predict values with your pre-trained gradient boosting regressors
        prediction = predict_with_model(embedded_text)

        # 4. Send response with results to client
        return jsonify({'prediction': prediction})

    @app.route('/')
    def hello_world():
        return 'Hello, World!'
    
    return app


if __name__ == '__main__':
    serve(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
