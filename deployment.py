

import pandas as pd
import pickle
import zipfile
from datetime import datetime, timedelta

def load_model(model_path, inside_zip_path=None):
    """Load a model from a given path, which could be a zip file or a pickle file."""
    # Check file extension
    if model_path.endswith('.zip'):
        # Handle zip file
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            if inside_zip_path:
                with zip_ref.open(inside_zip_path) as file:
                    return pickle.load(file)
            else:
                # Assuming there's only one file in the zip, automatically extract it
                with zip_ref.open(zip_ref.namelist()[0]) as file:
                    return pickle.load(file)
    else:
        # Handle regular pickle file
        with open(model_path, 'rb') as file:
            return pickle.load(file)

# Define predict_sample
def predict_sample(dataset_path, sample_size, adaboost_model, xgboost_model):
    """
    Loads a dataset, samples from it, and makes predictions using the provided models.

    :param dataset_path: Path to the dataset file.
    :param sample_size: Number of samples to take from the dataset.
    :param adaboost_model: The AdaBoost model for the first stage prediction.
    :param xgboost_model: The XGBoost model for the second stage prediction.
    :return: A DataFrame containing the predictions.
    """
    # Load the dataset
    X = pd.read_csv(dataset_path)

    # Sample from the dataset
    X_sampled = X.sample(n=sample_size) # no random state

    # Make predictions using the AdaBoost model
    X_sampled['adaboost_binary_pred'] = adaboost_model.predict(X_sampled)

    # Make predictions using the XGBoost model
    X_sampled['surface_area'] = xgboost_model.predict(X_sampled)
    
    # Adjust surface_area labels
    X_sampled['surface_area'] = X_sampled['surface_area'].map({0: 'BenignTraffic', 1:'Recon-HostDiscovery', 2:'Recon-OSScan',3:'Recon-PortScan',4:'VulnerabilityScan'})

    return X_sampled


# Load models
adaboost_loaded = load_model('pipeline_adaboost.zip')
xgboost_loaded = load_model('gridsearch_xgboost.pkl')

# Example usage
dataset_path = "X_test.csv"
sample_size = 50  
predictions = predict_sample(dataset_path, sample_size, adaboost_loaded, xgboost_loaded)

# Convert 'flow_duration' to timedelta
predictions['flow_duration_td'] = pd.to_timedelta(predictions['flow_duration'], unit='s')

# Calculate the cumulative sum of 'flow_duration' to simulate timestamps
predictions['time_stamp'] = predictions['flow_duration_td'].cumsum()

# get the current time
start_time = datetime.now()

# Add the cumulative sum to the start time to get the actual 'time_stamp'
predictions['time_stamp'] = predictions['time_stamp'].apply(lambda x: start_time + x)

# drop the 'flow_duration_td' column
predictions.drop(columns=['flow_duration_td'], inplace=True)

# Now, 'predictions' will have a 'time_stamp' column representing the time of each event

report = predictions[['surface_area','time_stamp']][predictions['surface_area'] != 'BenignTraffic']

report_dict = {
    category: datetime.utcfromtimestamp(time_stamp.astype('O')/1e9).strftime('%Y-%m-%d %H:%M:%S')
    for category, time_stamp in zip(report['surface_area'].values, report['time_stamp'].values)
}

print(report_dict)





    


    











