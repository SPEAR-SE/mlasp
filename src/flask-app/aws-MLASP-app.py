import random
from wtforms import SubmitField, StringField
from flask_wtf import FlaskForm
import joblib
import numpy as np
from flask import Flask, request, jsonify, session, url_for, render_template, redirect
from tensorflow.keras.models import load_model


def generateInputSequence(featSpace, exceptionList=None):
    """ 
    Iterates through the feature space dictionary and randomly selects a viable value to be added to the test feature
     object for each key in the dictionary.
    In case the key is found in the exceptionList, then it will randomly select  a value from the existing list,
     otherwise it will randomly select an element from the range applicable to the key.
    E.g.: featSpace = { key1:[min, max], key2:[value1, value2, value3]}, exceptionList=['key2'], then result will be a
     vector [random.randint(min, max), random.choice(value1, value2, value3)]

    Helper function used inside generateParameterCombinations function.
    
    Parameters:
        featSpace (dict): A dictionary defining the space of features of the model. The keys are the feature names and
         the values are applicable for the features. For a range, the values must be integers.
         For a choice, can be interger or float.
        exceptionList (list): The array of the feature names that should be used with random.choice(),
         i.e. multiple exact options possible instead of range.
        
    Returns:
        result(array): A feature vector, ordered as expected by the model.

    """
    if exceptionList is None:
        exceptionList = []
    result = []
    keys = list(featSpace)
    keys_len = len(featSpace.keys())
    for i in range(keys_len):
        val = 0
        if keys[i] in exceptionList:
            val = random.choice(featSpace.get(keys[i]))
        else:
            val = random.randint(featSpace[keys[i]][0], featSpace[keys[i]][1])

        result.append(val)

    return result


def generatePrediction(params, model, trainingScaler, targetScaler):
    """
    Generate a prediction for the given model using the provided list of params for the feature test values.
    Helper function used inside generateParameterCombinations function.
    
    Parameters:
        params(list): Ordered list (as expected by the model) of the test feature values.
        model(model): TF2 Keral trained model used to generate predictions.
        trainingScaler(scaler): Scaler for transforming the input parameters to the scale required by the model.
        targetScaler(scaler): Inverse scaler for transforming the prediction to actual scale.
    
    Returns:
        y_pred(array[float]): One dimensional array with the actual prediction (as float value).
    """
    params = np.array(params).reshape(1, -1)  # reshape from list to numpy array usable by the Keras scaler.
    scaledInput = trainingScaler.transform(params)
    y_pred_scaled = model.predict(scaledInput)
    y_pred = targetScaler.inverse_transform(y_pred_scaled)

    return y_pred


def generateParameterCombinations(featureSpace, exceptionList, epochs, precision, searchTarget, model, trainingScaler,
                                  targetScaler):
    """
    Searches for valid parameter combinations for a model within a given feature space, using a desired precision from
    the target prediction. The search is executed for a number of epochs.
    This is the entry level function that gets a full dictionary of possible parameter options that yield the search
    target (throughput).
    
    Parameters:
        featureSpace(dict): A dictionary defining the space of features of the model. The keys are the feature names and
         the values are applicable for the features. For a range, the values must be integers.
         For a choice, can be interger or float.
            E.g. featSpace = { key1:[min, max], key2:[value1, value2, value3]}, exceptionList=['key2'],
             then result will be a vector [random.randint(min, max), random.choice(value1, value2, value3)
        exceptionList (list): The array of the feature names that should be used with random.choice(),
         i.e. multiple exact options possible instead of range.
        epochs(int): The number of iterations to use random search for potential parameter combinations.
        precision(float): The precision (absolute deviation percentage) of the predicted target from the search target.
        searchTarget(int): The desired prediction value of the model for which a set of features are searched.
         It can also be a floating point number.
        model(model): The fully trained TF2 model loaded from an .h5 file.
        trainingScaler(scaler): The fitted scaler used before training or evaluation on the input data before passing it
         to the model. The scaler model is loaded from a pickle (.pkl) file.
        targetScaler(scaler): The fitted scaler of the target for the inverse transformation of the predicted value to
         the actual value that was used by the ML model during training and evaluation.
         The scaler model is loaded from a pickle (.pkl) file.
        
    Returns:
        parameters(dict): A dictionary containing lists of values for eligible parameters falling within the search
         patterns, their associated predictions and deviation measurements.
         The dictionaly keys are 'parameters', 'deviation' and 'predictions'
            E.g.: {'parameters': [[val_11, val_21, val_31, val_41, , val_n1],
                                  [val_12, val_22, val_32, val_42, , val_n2]],
                   'deviation': [array([[dev1]]),
                                  array([[dev2]])],
                   'predictions': [array([[pred1]], dtype=float32),
                                  array([[pred2]], dtype=float32)]}
    """
    numParams = len(featureSpace.keys())
    zeros = np.zeros(numParams)  # temp for initialization
    parameters = [zeros]
    deviation = [0]
    predictions = [0]

    print(f'Generating {epochs} sequences...\n')

    for i in range(epochs):
        inputSequence = generateInputSequence(featureSpace, exceptionList)
        y_pred = generatePrediction(inputSequence, model, trainingScaler, targetScaler)
        crt_dev = 100 * (abs(y_pred - searchTarget) / searchTarget)
        if crt_dev < precision:
            deviation.append(crt_dev)
            parameters.append(inputSequence)
            predictions.append(y_pred)

    parameters = parameters[1:]  # remove the first dummy element from all lists before creating the final output
    deviation = deviation[1:]
    predictions = predictions[1:]
    results = {'parameters': parameters, 'deviation': deviation, 'predictions': predictions}
    print(f'Done... Results are: {results} \n')
    return results


def extractBestParameterCombination(parameterCombinations):
    """
    Extracts the feature set from the input parameterCombinations dictionary created by the 'generateParameterCombinations'
     function closest to the search target (smallest deviation).
    
    Parameters:
        parameterCombinations(dict): A dictionary containing lists of values for eligible parameters falling within the
         search patterns, their associated predictions and deviation measurements. The dictionaly keys are 'parameters',
         'deviation' and 'predictions'
            E.g.: {'parameters': [[val_11, val_21, val_31, val_41, , val_n1],
                                  [val_12, val_22, val_32, val_42, , val_n2]],
                   'deviation': [array([[dev1]]),
                                  array([[dev2]])],
                   'predictions': [array([[pred1]], dtype=float32,
                                  array([[pred2]], dtype=float32}
        
    Returns:
        result(tuple): A three element tuple returning the parameters (a.k.a input features) array, the deviation percentage
         value from the searched target (as a float) and the predicted value for the given input set (as a float value).
    """

    parameters = parameterCombinations.get("parameters")
    deviation = parameterCombinations.get("deviation")
    predictions = parameterCombinations.get("predictions")

    pos = np.argmax(predictions)
    paramMap = {'BackgroundThreads': parameters[pos][0], 'LogCleanerThreads': parameters[pos][1],
                'NumIoThreads': parameters[pos][2], 'NumNetworkThreads': parameters[pos][3],
                'NumPartitions': parameters[pos][4], 'NumNodes': parameters[pos][5],
                'NumReplicaFetchers': parameters[pos][6], 'ThreadsClient': parameters[pos][7],
                'MessageSize': parameters[pos][8]
                }
    bestCombination = {'Parameters': paramMap,
                       'Deviation': deviation[pos][0][0],
                       'Prediction': float(predictions[pos][0][0])}
    return bestCombination


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def extractFeatureSpace(content):
    features = {'BackgroundThreads': list(map(num, content['BackgroundThreads'].split(','))),
                'LogCleanerThreads': list(map(num, content['LogCleanerThreads'].split(','))),
                'NumIoThreads': list(map(num, content['NumIoThreads'].split(','))),
                'NumNetworkThreads': list(map(num, content['NumNetworkThreads'].split(','))),
                'NumPartitions': list(map(num, content['NumPartitions'].split(','))),
                'NumNodes': list(map(num, content['NumNodes'].split(','))),
                'NumReplicaFetchers': list(map(num, content['NumReplicaFetchers'].split(','))),
                'ThreadsClient': list(map(num, content['ThreadsClient'].split(','))),
                'MessageSize': list(map(num, content['MessageSize'].split(',')))}
    # features['BackgroundThreads'] = np.fromstring(content['BackgroundThreads'], dtype=int, sep=',')
    print(f'Features are: {features}\n')
    return features


def extractExceptionList(features):
    exceptionList = []
    keys = list(features)
    keys_len = len(features.keys())
    for i in range(keys_len):
        # print(f'Key:{keys[i]}, feature:{features[keys[i]]}, length:{len(features[keys[i]])}')
        if len(features[keys[i]]) > 2:
            exceptionList.append(keys[i])
    print(f'Exception list is: {exceptionList}\n')
    return exceptionList


def extractEpochs(content):
    epochsValue = content['Epochs']
    epochs = num(epochsValue)
    return epochs


def extractPrecision(content):
    precisionValue = content['Precision']
    precision = num(precisionValue)
    return precision


def extractSearchTarget(content):
    searchTargetValue = content['SearchTargetValue']
    searchTarget = num(searchTargetValue)
    return searchTarget


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class AWSKafkaConfigForm(FlaskForm):
    BackgroundThreads = StringField('BackgroundThreads')
    LogCleanerThreads = StringField('LogCleanerThreads')
    NumIoThreads = StringField('NumIoThreads')
    NumNetworkThreads = StringField('NumNetworkThreads')
    NumPartitions = StringField('NumPartitions')
    NumNodes = StringField('NumNodes')
    NumReplicaFetchers = StringField('NumReplicaFetchers')
    ThreadsClient = StringField('ThreadsClient')
    MessageSize = StringField('MessageSize')
    Epochs = StringField('Epochs')
    Precision = StringField('Precision')
    SearchTargetValue = StringField('SearchTargetValue')
    submit = SubmitField('Submit')


@app.route("/", methods=["GET", "POST"])
def index():
    # return '<h1> Flask is running</h1>'
    form = AWSKafkaConfigForm()

    if form.validate_on_submit():
        session['BackgroundThreads'] = form.BackgroundThreads.data
        session['LogCleanerThreads'] = form.LogCleanerThreads.data
        session['NumIoThreads'] = form.NumIoThreads.data
        session['NumNetworkThreads'] = form.NumNetworkThreads.data
        session['NumPartitions'] = form.NumPartitions.data
        session['NumNodes'] = form.NumNodes.data
        session['NumReplicaFetchers'] = form.NumReplicaFetchers.data
        session['ThreadsClient'] = form.ThreadsClient.data
        session['MessageSize'] = form.MessageSize.data
        session['Epochs'] = form.Epochs.data
        session['Precision'] = form.Precision.data
        session['SearchTargetValue'] = form.SearchTargetValue.data
        return redirect(url_for("prediction"))
    return render_template('home.html', form=form)

##### !!!! Ensure the below files are in your folder ######
ml_model = load_model("ml_model_all.h5")
training_scaler = joblib.load("training_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")


@app.route("/prediction")
def prediction():
    results = "No valid combination found for given limits and search target value and precision"
    featureSpace = extractFeatureSpace(session)
    exceptionList = extractExceptionList(featureSpace)
    epochs = extractEpochs(session)
    precision = extractPrecision(session)
    searchTarget = extractSearchTarget(session)

    parameterCombinations = generateParameterCombinations(featureSpace, exceptionList, epochs, precision, searchTarget,
                                                          ml_model, training_scaler, target_scaler)
    if len(parameterCombinations.get("parameters")) == 0:
        print(f'No valid combination found for given limits and search target value and precision\n')
        # return "No valid combination found for given limits and search target value and precision"
    else:
        bestParamCombination = extractBestParameterCombination(parameterCombinations)
        print(f'Best parameter combination:{bestParamCombination}')
        # results = jsonify(bestParamCombination)
        results = bestParamCombination

    return render_template('prediction.html', results=results)


@app.route("/api/predict", methods=["POST"])
def ml_predict():
    content = request.json
    featureSpace = extractFeatureSpace(content)
    exceptionList = extractExceptionList(featureSpace)
    epochs = extractEpochs(content)
    precision = extractPrecision(content)
    searchTarget = extractSearchTarget(content)

    parameterCombinations = generateParameterCombinations(featureSpace, exceptionList, epochs, precision, searchTarget,
                                                          ml_model, training_scaler, target_scaler)
    if len(parameterCombinations.get("parameters")) == 0:
        print(f'No valid combination found for given limits and search target value and precision\n')
        return "No valid combination found for given limits and search target value and precision"
    else:
        bestParamCombination = extractBestParameterCombination(parameterCombinations)
        print(f'Best parameter combination:{bestParamCombination}')
        return jsonify(bestParamCombination)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
