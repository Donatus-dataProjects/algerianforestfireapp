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

#Styling my
    st.markdown("<h2 style='color: red; font-size: 16px;'>Input the required values:</h2>", unsafe_allow_html=True)

    # Collecting user input values from end user
    #Creating columns
    col1, col2, col3 = st.columns(3)
    with col1:
        Temperature = st.number_input("Temperature", min_value=-100.0, max_value=100.0, value=None, format="%.2f")
    
    with col2:
        RH = st.number_input("Relative Humidity", min_value=0.0, max_value=100.0, value=None, format="%.2f")
    
    with col3:
        Ws = st.number_input("Wind Speed", min_value=0.0, value=None, format="%.2f")
    
    with col1:
        Rain = st.number_input("Rain", min_value=0.0, value=None, format="%.2f")
    
    with col2:
        FFMC = st.number_input("Fine Fuel Moisture Code", min_value=0.0, value=None, format="%.2f")
    
    with col3:
        DMC = st.number_input("Duff Moisture Code", min_value=0.0, value=None, format="%.2f")
    
    with col1:
        ISI = st.number_input("Initial Spread Index", min_value=0.0, value=None, format="%.2f")
    
    with col2:
        Classes = st.selectbox("Classes (Select 0 or 1)", options=[0, 1])
    
    with col3:
        Region = st.selectbox("Region (Select 0 or 1)", options=[0, 1])

    # When user clicks "Predict"
    if st.button("Fire Weather Index"):
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