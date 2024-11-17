import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load prominent pollutants data
@st.cache_data
def load_pollutants_data():
    pollutants_df = pd.read_csv("ProminentPollutants.csv")  # Replace with actual file path
    return pollutants_df

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
        return "Invalid date format. Please use dd-mm-yyyy format.", None
    
    try:
        city_encoded = le_city.transform([user_city])[0]
    except ValueError:
        return f"City '{user_city}' not found in the training data.", None
    
    user_input = pd.DataFrame({'Year': [year], 'Month': [month], 'Day': [day], 'City': [city_encoded]})
    try:
        predicted_index_value = model_index.predict(user_input)
        return predicted_index_value[0], None
    except Exception as e:
        return f"Error during prediction: {e}", None

# Function to get prominent pollutants
def get_prominent_pollutants(city, pollutants_df):
    try:
        pollutants = pollutants_df[pollutants_df['City'] == city]['Prominent Pollutant'].values
        if len(pollutants) > 0:
            return pollutants[0]
        else:
            return "No data available."
    except Exception as e:
        return f"Error retrieving pollutants: {e}"

# Streamlit app layout
def main():
    st.title("Pollution Index Prediction")
    df, le_city = load_and_preprocess_data()
    pollutants_df = load_pollutants_data()
    model_index = train_model(df)
    
    st.sidebar.header("User Input")
    user_date = st.sidebar.text_input("Enter the date (dd-mm-yyyy):")
    user_city = st.sidebar.selectbox("Select City", le_city.classes_)
    
    if st.sidebar.button("Predict"):
        if user_date and user_city:
            predicted_index_value, error = predict_index(user_date, user_city, model_index, le_city)
            if error or isinstance(predicted_index_value, str):  # If there's an error
                st.error(predicted_index_value or error)
            else:
                st.success(f"Predicted Index Value: {predicted_index_value}")
                prominent_pollutants = get_prominent_pollutants(user_city, pollutants_df)
                st.info(f"Prominent Pollutants for {user_city}: {prominent_pollutants}")
        else:
            st.warning("Please enter both the date and city.")
    
    # Visualizations section
    st.header("City-Specific Data Visualizations")
    
    # Filter the dataframe by the selected city
    city_encoded = le_city.transform([user_city])[0]
    df_city = df[df['City'] == city_encoded]

    # 1. Pollution Trend by City (Line plot over time)
    st.subheader(f"Pollution Index Trend for {user_city}")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_city, x='Date', y='Index Value')
    plt.title(f"Pollution Index Trend in {user_city}")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # 2. City-wise Pollution Distribution (Box plot)
    st.subheader(f"Pollution Index Distribution in {user_city}")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_city, x='City', y='Index Value')
    plt.title(f"Pollution Index Distribution in {user_city}")
    st.pyplot(plt)

    # 3. Time Series of Pollution Index for the selected city
    st.subheader(f"Time Series of Pollution Index for {user_city}")
    df_grouped_by_date = df_city.groupby('Date')['Index Value'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_grouped_by_date, x='Date', y='Index Value')
    plt.title(f"Pollution Index Over Time in {user_city}")
    plt.xticks(rotation=45)
    st.pyplot(plt)

if __name__ == "__main__":
    main()
