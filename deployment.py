# Import the modules

import pandas as pd
import pickle

from datetime import datetime, timedelta
# Load the data




#with open('pipeline_adaboost.pkl', 'rb') as file:
    #adaboost_loaded = pickle.load(file)
    
#with open('grid_search_xgboost.pkl', 'rb') as file:
    #xgboost_loaded = pickle.load(file)





def load_model(model_path):
    """Load a model from a given path."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)



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
    X_sampled = X.sample(n=sample_size, random_state=42)

    # Make predictions using the AdaBoost model
    X_sampled['adaboost_binary_pred'] = adaboost_model.predict(X_sampled)

    # Make predictions using the XGBoost model
    X_sampled['surface_area'] = xgboost_model.predict(X_sampled)
    
    # Adjust surface_area labels
    X_sampled['surface_area'] = X_sampled['surface_area'].map({0: 'BenignTraffic', 1:'Recon-HostDiscovery', 2:'Recon-OSScan',3:'Recon-PortScan',4:'VulnerabilityScan'})

    return X_sampled



# Load models
adaboost_loaded = load_model('pipeline_adaboost.pkl')
xgboost_loaded = load_model('gridsearch_xgboost.pkl')

# Example usage
dataset_path = "X_test.csv"
sample_size = 50  
predictions = predict_sample(dataset_path, sample_size, adaboost_loaded, xgboost_loaded)


# Assuming 'predictions' is your DataFrame and it already has a 'flow_duration' column
# Let's also assume 'flow_duration' is in seconds for this example

# Convert 'flow_duration' to timedelta (adjust the unit if necessary, e.g., milliseconds)
predictions['flow_duration_td'] = pd.to_timedelta(predictions['flow_duration'], unit='s')

# Calculate the cumulative sum of 'flow_duration' to simulate timestamps
predictions['time_stamp'] = predictions['flow_duration_td'].cumsum()

# Assuming the execution starts now, get the current time
start_time = datetime.now()

# Add the cumulative sum to the start time to get the actual 'time_stamp'
predictions['time_stamp'] = predictions['time_stamp'].apply(lambda x: start_time + x)

# Optionally, you can drop the 'flow_duration_td' column if it's no longer needed
predictions.drop(columns=['flow_duration_td'], inplace=True)

# Now, 'predictions' will have a 'time_stamp' column representing the time of each event



report = predictions[['surface_area','time_stamp']][predictions['surface_area'] != 'BenignTraffic']

report_dict = {str(t): a for a, t in [*zip(predictions['time_stamp'].values, predictions['surface_area'].values)]}

print(report_dict)





    


    











