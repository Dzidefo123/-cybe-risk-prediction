import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Load the saved data
with open('saved_steps1.pkl', 'rb') as file:
    data = pickle.load(file)
    
model_loaded = data['model']
le_industry = data['le_industry']
le_threats = data['le_threats']
le_vulnerabilities = data['le_vulnerabilities']
le_information_value = data['le_information_value']


# Dictionary to store threat-specific recommendations
threat_recommendations = {
    'ransomware': ['Security Software', 'Email and Web Safety'],
    'DDoS': ['DDoS Protection Services', 'Network Security'],
    'malware': ['Enable Firewall Protection', 'Use Reputable Security Software'],
    'Data_breaches': ['Access Control', 'Encryption'],
    'Man_in_the_middle_attack': ['Use Secure Protocols', 'Verify SSL/TLS Certificates']
}

def calculate_premium(predicted_prob, threshold):
    if predicted_prob > threshold:
        return 10000
    elif 0.40 <= predicted_prob <= threshold:
        return 7000
    else:
        return 4000

st.sidebar.image('3D_Animation_Style_CAT_EYE_123.jpg', width=500)


def main():
    st.title('Cyber Risk Probability simulation & Fraud Prvention App')

    # Create interactive widgets
    industry_encoded = st.selectbox('Select Industry', le_industry.classes_)
    threats_encoded = st.selectbox('Select Threats', le_threats.classes_)
    vulnerabilities_encoded = st.selectbox('Select Vulnerabilities', le_vulnerabilities.classes_)
    information_value_encoded = st.selectbox('Select Information Value', le_information_value.classes_)

    if st.button('Calculate Probability'):
        # Create input array for prediction
        input_data = np.array([
            le_industry.transform([industry_encoded])[0],
            le_threats.transform([threats_encoded])[0],
            le_vulnerabilities.transform([vulnerabilities_encoded])[0],
            le_information_value.transform([information_value_encoded])[0]
        ]).reshape(1, -1)

        # Make prediction using the loaded model
        predicted_class = model_loaded.predict(input_data)
        predicted_prob = model_loaded.predict_proba(input_data)[0][1]  # Probability of Risks
        
        st.subheader('Premium Calculation')
        threshold = st.slider('Select Probability Threshold:', 0.0, 1.0, 0.7, 0.01)
        premium = calculate_premium(predicted_prob, threshold)
        st.write(f'Probability of Risk: {predicted_prob:.2f}')
        st.write(f'Premium per Month: {premium} BRL')

        st.write(f'Probability of Risk: {predicted_prob:.2f}')
        
        # Display threat-specific recommendations based on the selected threat
    st.sidebar.subheader('Threat Recommendations')
    for threat_label in threat_recommendations:
        st.sidebar.write(f'For "{threat_label}" threat, consider these recommendations:')
        for recommendation in threat_recommendations[threat_label]:
            st.sidebar.write(f'- {recommendation}')    

        
      
        
 
    st.subheader('About the Author')
    st.markdown(
        "This simulation is created by Dzidefo Alomenu, a passionate data scientist and developer. Connect with Dzidefo "
        "on https://www.linkedin.com/in/dzidefo-alomenu-6772b733/ and https://medium.com/@calormenu explore more projects on his portfolio."
    )

    
        
        
     

if __name__ == '__main__':
    main()
