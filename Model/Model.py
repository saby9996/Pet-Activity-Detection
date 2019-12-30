# Developed By Code|<Ill at 7/8/2019
# Developed VM IP 203.241.246.158

#Imports Model Load
import pickle
from scipy.stats import mode
from Utilities import helper_functions



def Model_Activity():
    model_path='Model/Activity_Model.pkl'
    model_file=open(model_path,'rb')
    model=pickle.load(model_file)

    return model


def model_call(data):
    feature_set=['N-AccX', 'N-AccY', 'N-AccZ', 'N-GyroX', 'N-GyroY', 'N-GyroZ', 'T-AccX', 'T-AccY', 'T-AccZ', 'T-GyroX', 'T-GyroY', 'T-GyroZ']
    Label=data['Label'].mode()[0]
    features=data[feature_set]
    model=Model_Activity()
    predictions=model.predict(features)

    data['Predictions']=predictions
    data['Information']=Label+'-Activity'
    data['Predictions']=helper_functions.Num_to_Label(data['Predictions'])

    return data





