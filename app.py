import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


st.write("""
         # Weight prediction
         """
         )
data = {
        'claw_length_cm': [1, 2, 3, 4, 5],
        'endangered': [0, 1, 0, 1, 0],
        'size_cm': [10, 15, 20, 25, 30],
        'specie': [0, 1, 0, 1, 0],
        'sub_specie': [0, 1, 0, 1, 0],
        'tail_length_cm': [5, 10, 15, 20, 25],
        'weight_kg': [2, 4, 6, 8, 10]
}
df = pd.DataFrame(data)


#df = pd.read_csv("sloth_data.csv")
X = df[['claw_length_cm', 'endangered', 'size_cm', 'specie', 'sub_specie', 'tail_length_cm']]
y = df['weight_kg']


claw_length = st.number_input('Claw Length (cm)', min_value=0.0)
endangered = st.selectbox('Endangered', df['endangered'].unique())
size = st.number_input('Size (cm)', min_value=0.0)
specie = st.selectbox('Specie', df['specie'].unique())
sub_specie = st.selectbox('Sub Specie', df['sub_specie'].unique())
tail_length = st.number_input('Tail Length (cm)', min_value=0.0)

input_data = pd.DataFrame({'claw_length_cm': [claw_length],
                           'endangered': [endangered],
                           'size_cm': [size],
                           'specie': [specie],
                           'sub_specie': [sub_specie],
                           'tail_length_cm': [tail_length]})

label_encoder=LabelEncoder()
categorical_columns=input_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
  input_data[col]=label_encoder.fit_transform(input_data[col])


model = RandomForestRegressor(max_depth=16, random_state=0)
model.fit(X_train, y_train)

weight_prediction = model.predict(input_data)

st.subheader('Súlybecslés eredménye:')
st.write(f'A becsült súly: {weight_prediction[0]:.2f} kg')

