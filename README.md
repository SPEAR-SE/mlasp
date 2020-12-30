# MLASP - Machine Learning Assisted System Performance and Capacity Prediction Based on Configuration Parameter Values
This repository contains the open-source system experimental data (code and test data) used as part of a research project conducted by members of the SPEAR lab at Concordia University, Montreal, Quebec, Canada.

The provided source code includes the feature selection and cleanup steps required before feeding the input data to various ML algorithms.
Note there are independent CSV files with raw data which are processed using different feature engineering techniques to create the data sets for model trainings.
The feature space (from the CSV files) is as follows (with full details provided in the paper TODO: add link): 
- Kafka broker configuration parameters: 'BackgroundThreads', 'LogCleanerThreads', 'NumIoThreads', 'NumNetworkThreads', and 'NumReplicaFetchers' (see Apache Kafka documentation for details)
- Kafka cluster parameters: 'NumNodes' (the number of nodes in the Kafka cluster)
- Kafka topic parameters: 'NumPartitions' (for the number or partitions for a kafka topic)
- Kafka producer (client) parameters (used to control the load generation to Kafka): 'ThreadsClient' (number of threads writing to Kafka), and 'MessageSize' (the size of the message sent to Kafka).

The target variable used in model prediction is 'TotalMessages' (as the number of messages recorded by Kafka over the time of load test).

The repository also contains a Flask based model application for how a trained model may be used for finding a configuration setting that may produce a desired target value. The flask app uses random values from a search space interval to generate configurations and measure the prediction against a designated target with a specified error margin. The number of search iteration as well as the error margin are part of the search criteria. The app exposes both a UI and REST API interface (accepting POST requests with JSON payload). To test for the defaults (i.e. http://localhost:5000/api/predict) , the following query may be initiated (i.e., using Postman for testing purposes):
```json
{
 "BackgroundThreads": "5, 30",
 "LogCleanerThreads": "1, 1",
 "NumIoThreads": "4, 16",
 "NumNetworkThreads": "3, 6",
 "NumPartitions": "1, 2",
 "NumNodes": "1, 2",
 "NumReplicaFetchers": "1, 2",
 "ThreadsClient": "2.30258509, 2.7080502, 2.99573227, 3.21887582, 3.91202301",
 "MessageSize": "10240, 10240",
 "Epochs": "100",
 "SearchTargetValue": "500000",
 "Precision": "3.9"
}
```
Note that in case of the Flask app, ranges are applicable for the feature values only when two values are provided (i.e. BackgroundThreads will have random integer values between 5 and 30), and a choice of possible values is used when multiple entries are provided (i.e. ThreadsClient will have one of the given samples - note that ThreadsClient here is using normalized values, that's why they are float numbers).

NOTES:
- The source code and test data is provided only for the open source system
- The source code is in Jupyter notebook format
- The notebooks expect to have an environment with all dependencies pre-installed.
- The batch run notebook has hardcoded the value of the Jupyter virtual environment to tf-cpu. Please adjust according to your environment settings.
- The Flask sample app assumes using an NN model, it may however be easily adapted to use any model. Please ensure you use the Flask app with the same python virtual environment settings (libraries) as used during training of the models.
- To use the Flask app, an NN model needs to be trained and the corresponding .h5 model file and .pkl for the scalers (feature space and target) need to be placed in the app folder

The contents of this repository are provided "as-is" without further support. In other words, it is expected of the user to have sufficient Python (and ML) related knowledge to understand what the code does and how it works.



