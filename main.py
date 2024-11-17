import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load and clean the data
@st.cache_data  # Cache the data loading for faster reloads
def load_and_preprocess_data():
    df = pd.read_csv("Data.csv")  # Replace with actual file path
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    le_city = LabelEncoder()
    df['City'] = le_city.fit_transform(df['City'])
    return df, le_city

# Train the model
@st.cache_resource  # Cache the model to avoid retraining each time
def train_model(df):
    X = df[['Year', 'Month', 'Day', 'City']]
    y = df['Index Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_index = RandomForestRegressor()
    model_index.fit(X_train, y_train)
    joblib.dump(model_index, 'model_index.pkl')
    return model_index

# Function to make predictions
def predict_index(user_date, user_city, model_index, le_city):
    try:
        user_date = datetime.strptime(user_date, '%d-%m-%Y')
        year, month, day = user_date.year, user_date.month, user_date.day
    except ValueError:
        return "Invalid date format. Please use dd-mm-yyyy format."
    
    try:
        city_encoded = le_city.transform([user_city])[0]
    except ValueError:
        return f"City '{user_city}' not found in the training data."
    
    user_input = pd.DataFrame({'Year': [year], 'Month': [month], 'Day': [day], 'City': [city_encoded]})
    try:
        predicted_index_value = model_index.predict(user_input)
        return predicted_index_value[0]
    except Exception as e:
        return f"Error during prediction: {e}"

# Streamlit app layout
def main():
    st.title("Pollution Index Prediction")
    df, le_city = load_and_preprocess_data()
    model_index = train_model(df)
    st.sidebar.header("User Input")
    user_date = st.sidebar.text_input("Enter the date (dd-mm-yyyy):")
    user_city = st.sidebar.selectbox("Select City", le_city.classes_)
    
    if st.sidebar.button("Predict"):
        if user_date and user_city:
            predicted_index_value = predict_index(user_date, user_city, model_index, le_city)
            if isinstance(predicted_index_value, str):  # If there's an error
                st.error(predicted_index_value)
            else:
                st.success(f"Predicted Index Value: {predicted_index_value}")
        else:
            st.warning("Please enter both the date and city.")

if __name__ == "__main__":
    main()
