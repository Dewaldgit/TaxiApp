#### IMPORTS ####
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#### BUILDING CONTAINERS FOR SITE ####

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

### STYLING THE DATA ###
st.markdown(
    """
    <style>
    .main {
    background-color: #FFFFFF;
    }
    </style>
    }
    """,
    unsafe_allow_html=True
)

### FUNCTION TO CACHE THE DATA ###
@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename) 
    return taxi_data

#### BODY OF SITE ####

with header:
    st.title('Welcome to my Data Science Project')
    st.text('In this project I look at the tranactions made by taxis')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset at the New York City Taxi & Limousine Commision website.')
    st.text('https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page')

    # Importing the CSV data via Pandas
    taxi_data = get_data('Data/yellow_tripdata_2021-01.csv')
    st.write(taxi_data.head())

    st.subheader('Pick-up location ID distribution on the NYC dataset')
    pickup_location = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pickup_location)

with features:
    st.header('The features that I created')

    st.markdown('* **Feature One** I created this feature to highlight the feature')
    st.markdown('* **Feature Two** I created this feature to highlight the feature')

with model_training:
    st.header('Time to train the model')    
    st.text('Here you get to choose your hyperparamaters')

    # Building a Slider, Selecting Box and Text Input Box 
    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be', options=[100, 200, 300, 'No Limit'], index=0)

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input('Which feature should be used as an input?', 'PULocationID')

    # Building a Model - Random Forest Model

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared error of the model is:')
    disp_col.write(r2_score(y, prediction))