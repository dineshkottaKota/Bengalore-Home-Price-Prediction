import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
pickle_in = open("banglore_home_prices_model.pickle","rb")
classifier=pickle.load(pickle_in)

with open("columns.json", "r") as f:
    __data_columns = json.load(f)['data_columns']
    __locations = __data_columns[3:]

    
def welcome():
    return "Welcome All"



def predict_note_authentication(sqft,bhk,bath,loc):
    try:
        loc_index = __data_columns.index(loc.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    prediction=round(classifier.predict([x])[0],2)
    return round(classifier.predict([x])[0],2)

def main():
    html_temp = """
    <div style="padding:10px">
    <h2 style="color:white;text-align:center;">Banglore Home Price Predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sqft = st.text_input("Area (Total Square Feet)","")
    bhk = st.selectbox("BHK",('1','2','3','4','5'))
    bath = st.selectbox("Bath",('1','2','3','4','5'))
    loc = st.selectbox("Location",__locations)
    result=""
    if st.button("Estimate Price"):
        result=predict_note_authentication(sqft,bhk,bath,loc)
    st.success('Estimated Price is {} lakhs'.format(result))
    
if __name__=='__main__':
    main()
    
    
    
