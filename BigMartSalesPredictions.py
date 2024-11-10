import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import pickle
import streamlit as st

# Data Collection and Processing
big_mart_data = pd.read_csv('Train.csv')

# Handling Missing Values without inplace=True
big_mart_data['Item_Weight'] = big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean())

mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
miss_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size[x])

# Data Pre-Processing
big_mart_data = big_mart_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}})

# Initializing and fitting encoders
def encode_and_save(column_name, data, filename):
    encoder = LabelEncoder()
    data[column_name] = encoder.fit_transform(data[column_name])
    with open(filename, 'wb') as file:
        pickle.dump(encoder, file)
    return data

# Encoding each categorical column and saving encoders
big_mart_data = encode_and_save('Item_Identifier', big_mart_data, 'item_identifier_encoder.pkl')
big_mart_data = encode_and_save('Item_Fat_Content', big_mart_data, 'item_fat_content_encoder.pkl')
big_mart_data = encode_and_save('Item_Type', big_mart_data, 'item_type_encoder.pkl')
big_mart_data = encode_and_save('Outlet_Identifier', big_mart_data, 'outlet_identifier_encoder.pkl')
big_mart_data = encode_and_save('Outlet_Size', big_mart_data, 'outlet_size_encoder.pkl')
big_mart_data = encode_and_save('Outlet_Location_Type', big_mart_data, 'outlet_location_type_encoder.pkl')
big_mart_data = encode_and_save('Outlet_Type', big_mart_data, 'outlet_type_encoder.pkl')

# Splitting features and target
X = big_mart_data.drop(columns='Item_Outlet_Sales')
Y = big_mart_data['Item_Outlet_Sales']

# Splitting the data into Training and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Save the trained model
with open('big_mart_sales_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)

# Streamlit app
st.title("Big Mart Sales Prediction")

# Load encoders
def load_encoder(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

item_identifier_encoder = load_encoder('item_identifier_encoder.pkl')
item_fat_content_encoder = load_encoder('item_fat_content_encoder.pkl')
item_type_encoder = load_encoder('item_type_encoder.pkl')
outlet_identifier_encoder = load_encoder('outlet_identifier_encoder.pkl')
outlet_size_encoder = load_encoder('outlet_size_encoder.pkl')
outlet_location_type_encoder = load_encoder('outlet_location_type_encoder.pkl')
outlet_type_encoder = load_encoder('outlet_type_encoder.pkl')

# Collect user inputs
item_weight = st.number_input("Item Weight", min_value=0.0, step=0.1)
item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, step=0.01)
item_mrp = st.number_input("Item MRP", min_value=0.0, step=0.1)
outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2023, step=1)

# Categorical inputs
item_identifier = st.selectbox("Item Identifier", item_identifier_encoder.classes_)
item_fat_content = st.selectbox("Item Fat Content", item_fat_content_encoder.classes_)
item_type = st.selectbox("Item Type", item_type_encoder.classes_)
outlet_identifier = st.selectbox("Outlet Identifier", outlet_identifier_encoder.classes_)
outlet_size = st.selectbox("Outlet Size", outlet_size_encoder.classes_)
outlet_location_type = st.selectbox("Outlet Location Type", outlet_location_type_encoder.classes_)
outlet_type = st.selectbox("Outlet Type", outlet_type_encoder.classes_)

# Create a DataFrame for input data
input_data = pd.DataFrame({
    'Item_Identifier': [item_identifier],
    'Item_Weight': [item_weight],
    'Item_Fat_Content': [item_fat_content],
    'Item_Visibility': [item_visibility],
    'Item_Type': [item_type],
    'Item_MRP': [item_mrp],
    'Outlet_Identifier': [outlet_identifier],
    'Outlet_Establishment_Year': [outlet_establishment_year],
    'Outlet_Size': [outlet_size],
    'Outlet_Location_Type': [outlet_location_type],
    'Outlet_Type': [outlet_type]
})

# Encode categorical inputs
input_data['Item_Identifier'] = item_identifier_encoder.transform(input_data['Item_Identifier'])
input_data['Item_Fat_Content'] = item_fat_content_encoder.transform(input_data['Item_Fat_Content'])
input_data['Item_Type'] = item_type_encoder.transform(input_data['Item_Type'])
input_data['Outlet_Identifier'] = outlet_identifier_encoder.transform(input_data['Outlet_Identifier'])
input_data['Outlet_Size'] = outlet_size_encoder.transform(input_data['Outlet_Size'])
input_data['Outlet_Location_Type'] = outlet_location_type_encoder.transform(input_data['Outlet_Location_Type'])
input_data['Outlet_Type'] = outlet_type_encoder.transform(input_data['Outlet_Type'])

# Load the trained model
with open('big_mart_sales_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Predicted Sales:", prediction[0])
