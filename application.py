# coding=utf-8

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from base64 import b64encode, b64decode
from google.cloud import automl_v1beta1 as automl
import re

application = Flask(__name__)
CORS(application)


def predict(image):
    # TODO(developer): Uncomment and set the following variables
    project_id = 'qualiscan-216706'
    compute_region = 'us-central1'
    model_id = 'ICN2956512205565128229'
    # file_path = '/local/path/to/file'
    score_threshold = '0.5'

    automl_client = automl.AutoMlClient()

    # Get the full path of the model.
    model_full_id = automl_client.model_path(
        project_id, compute_region, model_id
    )

    # Create client for prediction service.
    prediction_client = automl.PredictionServiceClient()
    img_data = b64decode(image)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(img_data)

    # Read the image and assign to payload.
    with open(filename, "rb") as image_file:
        content = image_file.read()
    payload = {"image": {"image_bytes": content}}

    # params is additional domain-specific parameters.
    # score_threshold is used to filter the result
    # Initialize params
    params = {}
    if score_threshold:
        params = {"score_threshold": score_threshold}

    response = prediction_client.predict(model_full_id, payload, params)
    print("Prediction results:")
    results = []
    for result in response.payload:
        print(result)
        results.append({"display_name": result.display_name,
                        "classification_score": result.classification.score})
        # print("Predicted class name: {}".format(result.display_name))
        # print("Predicted class score: {}".format(result.classification.score))
    # print(response)
    return results


@application.route('/', methods=['GET', 'POST'])
@cross_origin(allow_headers=["Content-Type", "Connection", "x-auth", "x-key"])
def get_passport_data_results():
    return jsonify({
                    "message": "Hello world",
                    })


@application.route('/predict', methods=['GET', 'POST'])
@cross_origin(allow_headers=["Content-Type", "Connection", "x-auth", "x-key"])
def predict_disease():
    if request.method == 'POST':
        data = request.get_json()
        image = data["image"]
        image = re.sub('^data:image/.+;base64,', '', image)
        image = image.encode()
        print(image[:100])
        print(type(image))
        prediction_results = predict(image)
        return jsonify(prediction_results)


if __name__ == '__main__':
    application.run(threaded=True, debug=False)