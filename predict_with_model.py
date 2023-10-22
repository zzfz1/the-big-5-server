import joblib
import os
import numpy as np

def predict_with_model(input_data):
    """
    Predict the values using the saved models.

    Parameters:
    - input_data: The input data for prediction. It should be preprocessed and in the format expected by the models.

    Returns:
    - A list containing the predicted values for n, e, o, a, and c.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Load the saved models
    gb_n = joblib.load(os.path.join(script_dir,'gb_n_model.pkl'))
    gb_e = joblib.load(os.path.join(script_dir,'gb_e_model.pkl'))
    gb_o = joblib.load(os.path.join(script_dir,'gb_o_model.pkl'))
    gb_a = joblib.load(os.path.join(script_dir,'gb_a_model.pkl'))
    gb_c = joblib.load(os.path.join(script_dir,'gb_c_model.pkl'))
    
    # Make predictions using the loaded models
    pred_n = np.clip(gb_n.predict(input_data), 1, 5)
    pred_e = np.clip(gb_e.predict(input_data), 1, 5)
    pred_o = np.clip(gb_o.predict(input_data), 1, 5)
    pred_a = np.clip(gb_a.predict(input_data), 1, 5)
    pred_c = np.clip(gb_c.predict(input_data), 1, 5)
    
    # Return the predictions as a list
    return [pred_n[0], pred_e[0], pred_o[0], pred_a[0], pred_c[0]]


# from import_bert import import_bert
# import pandas as pd
# # Example usage:
# input_data_sample = import_bert(pd.DataFrame({'text':["self aware observant sympathetic kind intellectual"] }), 'text')
# predictions = predict_with_model(input_data_sample)
# print(predictions)
