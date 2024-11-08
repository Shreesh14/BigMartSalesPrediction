
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

"""Data Collection and Processing"""

# loading the data from csv file to Pandas DataFrame
big_mart_data = pd.read_csv(r'Train.csv')


# first 5 rows of the dataframe
big_mart_data.head()

# number of data points & number of features
big_mart_data.shape

# getting some information about thye dataset
big_mart_data.info()

"""Categorical Features:
- Item_Identifier
- Item_Fat_Content
- Item_Type
- Outlet_Identifier
- Outlet_Size
- Outlet_Location_Type
- Outlet_Type
"""

# checking for missing values
big_mart_data.isnull().sum()

"""Handling Missing Values

Mean --> average

Mode --> more repeated value
"""

# mean value of "Item_Weight" column
big_mart_data['Item_Weight'].mean()

# filling the missing values in "Item_weight column" with "Mean" value
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

# mode of "Outlet_Size" column
big_mart_data['Outlet_Size'].mode()

# filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

print(mode_of_Outlet_size)

miss_values = big_mart_data['Outlet_Size'].isnull()

print(miss_values)

big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

# checking for missing values
big_mart_data.isnull().sum()

"""Data Analysis"""

big_mart_data.describe()

"""Numerical Features"""

sns.set()

# Item_Weight distribution
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_Weight'])
plt.show()

# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_Visibility'])
plt.show()

# Item MRP distribution
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_MRP'])
plt.show()

# Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_Outlet_Sales'])
plt.show()

# Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()

"""Categorical Features"""

# Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

# Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

# Outlet_Size column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.show()

"""Data Pre-Processing"""

big_mart_data.head()

big_mart_data['Item_Fat_Content'].value_counts()

big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

big_mart_data['Item_Fat_Content'].value_counts()

"""Label Encoding"""

encoder = LabelEncoder()

big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])

big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])

big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])

big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])

big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])

big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])

big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

big_mart_data.head()

"""Splitting features and Target"""

X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

print(X)

print(Y)

"""Splitting the data into Training data & Testing Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Machine Learning Model Training

XGBoost Regressor
"""

regressor = XGBRegressor()

regressor.fit(X_train, Y_train)

"""Evaluation"""

# prediction on training data
training_data_prediction = regressor.predict(X_train)

# R squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)

print('R Squared value = ', r2_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)

# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)

print('R Squared value = ', r2_test)

model= regressor

import pickle
import streamlit as st
# Load the trained model
with open('big_mart_sales_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load each encoder
with open('item_fat_content_encoder.pkl', 'rb') as file:
    item_fat_content_encoder = pickle.load(file)
with open('item_type_encoder.pkl', 'rb') as file:
    item_type_encoder = pickle.load(file)
with open('outlet_identifier_encoder.pkl', 'rb') as file:
    outlet_identifier_encoder = pickle.load(file)
with open('outlet_size_encoder.pkl', 'rb') as file:
    outlet_size_encoder = pickle.load(file)
with open('outlet_location_type_encoder.pkl', 'rb') as file:
    outlet_location_type_encoder = pickle.load(file)
with open('outlet_type_encoder.pkl', 'rb') as file:
    outlet_type_encoder = pickle.load(file)

# Streamlit app
st.title("Big Mart Sales Prediction")

# Collect user inputs
item_weight = st.number_input("Item Weight", min_value=0.0, step=0.1)
item_fat_content = st.selectbox("Item Fat Content", item_fat_content_encoder.inverse_transform(range(len(item_fat_content_encoder.classes_))))
item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, step=0.01)
item_type = st.selectbox("Item Type", item_type_encoder.inverse_transform(range(len(item_type_encoder.classes_))))
item_mrp = st.number_input("Item MRP", min_value=0.0, step=0.1)
outlet_identifier = st.selectbox("Outlet Identifier", outlet_identifier_encoder.inverse_transform(range(len(outlet_identifier_encoder.classes_))))
outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1985, step=1)
outlet_size = st.selectbox("Outlet Size", outlet_size_encoder.inverse_transform(range(len(outlet_size_encoder.classes_))))
outlet_location_type = st.selectbox("Outlet Location Type", outlet_location_type_encoder.inverse_transform(range(len(outlet_location_type_encoder.classes_))))
outlet_type = st.selectbox("Outlet Type", outlet_type_encoder.inverse_transform(range(len(outlet_type_encoder.classes_))))

# Create a DataFrame for input data
input_data = pd.DataFrame({
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
input_data['Item_Fat_Content'] = item_fat_content_encoder.transform(input_data['Item_Fat_Content'])
input_data['Item_Type'] = item_type_encoder.transform(input_data['Item_Type'])
input_data['Outlet_Identifier'] = outlet_identifier_encoder.transform(input_data['Outlet_Identifier'])
input_data['Outlet_Size'] = outlet_size_encoder.transform(input_data['Outlet_Size'])
input_data['Outlet_Location_Type'] = outlet_location_type_encoder.transform(input_data['Outlet_Location_Type'])
input_data['Outlet_Type'] = outlet_type_encoder.transform(input_data['Outlet_Type'])

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Predicted Sales:", prediction[0])
