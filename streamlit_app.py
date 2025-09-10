import streamlit as st
import requests
import json

st.set_page_config(page_title='Obesity Prediction App', layout='centered')

st.title('🏥 Obesity Prediction App')
st.info('This app predicts the likelihood of obesity based on your lifestyle and habits.')

st.markdown('---')
st.header('🔍 Personal Information')

# Categorical Inputs - Personal & Lifestyle
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', min_value=0, max_value=100, value=25)

st.markdown('---')
st.header('🍔 Lifestyle Habits')

family_history_with_overweight = st.selectbox('Family History With Overweight', ['yes', 'no'])
favc = st.selectbox('Frequent High Caloric Food Consumption (FAVC)', ['yes', 'no'])
caec = st.selectbox('Consumption of Food Between Meals (CAEC)', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('Smoke', ['yes', 'no'])
scc = st.selectbox('Monitoring Calorie Consumption (SCC)', ['no', 'yes'])
calc = st.selectbox('Consumption of Alcohol (CALC)', ['no', 'Sometimes', 'Frequently'])
mtrans = st.selectbox('Transportation Used (MTRANS)', ['Automobile', 'Public Transportation', 'Walking', 'Bike', 'Motorbike'])

st.markdown('---')
st.header('⚙️ Body Metrics and Daily Habits')

height = st.slider('Height (cm)', min_value=100, max_value=250, value=170)
weight = st.slider('Weight (kg)', min_value=30, max_value=200, value=70)
fcvc = st.slider('Frequency of Vegetable Consumption (FCVC)', min_value=1, max_value=3, value=2)
ncp = st.slider('Number of Main Meals (NCP)', min_value=1, max_value=4, value=3)
ch2o = st.slider('Daily Water Consumption (CH2O)', min_value=1, max_value=3, value=2)
faf = st.slider('Physical Activity Frequency (FAF)', min_value=0, max_value=3, value=1)
tue = st.slider('Time Using Technology (TUE)', min_value=0, max_value=2, value=1)

inputs = {
    'Gender': gender,
    'family_history_with_overweight': family_history_with_overweight,
    'FAVC': favc,
    'CAEC': caec,
    'SMOKE': smoke,
    'SCC': scc,
    'CALC': calc,
    'MTRANS': mtrans,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'FCVC': fcvc,
    'NCP': ncp,
    'CH2O': ch2o,
    'FAF': faf,
    'TUE': tue
}

st.markdown('---')
if st.button('🚀 Predict'):
    with st.spinner('🔎 Predicting... Please wait...'):
        try:
            response = requests.post(
                url="http://127.0.0.1:8000/predict",
                data=json.dumps(inputs),
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result.get('predicted_obesity_level', None)

                if prediction:
                    # 🎈 Add balloon animation
                    st.balloons()

                    # 🎨 Color mapping for predictions
                    color_map = {
                        'Insufficient_Weight': 'blue',
                        'Normal_Weight': 'green',
                        'Overweight_Level_I': 'orange',
                        'Overweight_Level_II': 'darkorange',
                        'Obesity_Type_I': 'red',
                        'Obesity_Type_II': 'darkred',
                        'Obesity_Type_III': 'purple'
                    }

                    # Get color or default to black
                    color = color_map.get(prediction, 'black')

                    # 💡 Clean label format
                    clean_label = prediction.replace('_', ' ')

                    # 🎯 Show prediction result nicely
                    st.success('🎯 Prediction Result:')
                    st.markdown(f"""
                        <h2 style='text-align: center; color: {color};'>
                            {clean_label}
                        </h2>
                        """, unsafe_allow_html=True)

                else:
                    st.warning('⚠️ No prediction returned by the API.')

            else:
                st.error(f'❌ API Error: {response.text}')

        except Exception as e:
            st.error(f'❌ An unexpected error occurred: {str(e)}')
