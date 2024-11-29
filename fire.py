import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Loading the models
loaded_model = pickle.load(open('models/trained_model.sav', 'rb'))
standard_scaler = pickle.load(open('models/scaler.sav', 'rb'))

###Styling
# Streamlit app
def main():
    st.title("Forest Fire Prediction")

    st.header("Input the required values:")

    # Collecting user input values      
    Temperature = st.number_input("Temperature", min_value=-100.0, max_value=100.0, value=None, format="%.2f")
    RH = st.number_input("Relative Humidity (RH)", min_value=0.0, max_value=100.0, value=None, format="%.2f")
    Ws = st.number_input("Wind Speed (Ws)", min_value=0.0, value=None, format="%.2f")
    Rain = st.number_input("Rain", min_value=0.0, value=None, format="%.2f")
    FFMC = st.number_input("FFMC", min_value=0.0, value=None, format="%.2f")
    DMC = st.number_input("DMC", min_value=0.0, value=None, format="%.2f")
    ISI = st.number_input("ISI", min_value=0.0, value=None, format="%.2f")
    #Classes = st.number_input("Classes", min_value=0, max_value=100, value=None)
    #Region = st.number_input("Region", min_value=0, max_value=100, value=None)
    Classes = st.selectbox("Classes (Select 0 or 1)", options=[0, 1])
    Region = st.selectbox("Region (Select 0 or 1)", options=[0, 1])


    # When user clicks "Predict"
    # When user clicks "Predict"
    if st.button("Predict"):
        if None in [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]:
            st.error("Please fill in all fields before predicting.")
        else:
            # Prepare the input data
            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            
            # Scale input data and predict
            try:
                new_data_scaled = standard_scaler.transform(input_data)
                result = loaded_model.predict(new_data_scaled)
                st.success(f"Prediction: {result[0]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Running the main function
if __name__ == "__main__":
    main()