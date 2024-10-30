import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load and prepare data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('Crop_recommendation.csv')
        st.sidebar.success("Dataset loaded successfully!")
        return data
    except FileNotFoundError:
        st.error("Error: 'Crop_recommendation.csv' file not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load data
data = load_data()

if data is not None:
    # Display data info in sidebar
    st.sidebar.write("Dataset Information:")
    st.sidebar.write(f"Total samples: {len(data)}")
    st.sidebar.write(f"Features available: {list(data.columns)}")

    # Main navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Crop Prediction", "Model Analysis"])

    if page == "Data Overview":
        st.title("📊 Crop Dataset Overview")
        
        # Show first few rows
        st.subheader("Sample Data")
        st.write(data.head())
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.write(data.describe())
        
        # Data visualizations
        st.subheader("Data Visualizations")
        
        # Select columns for visualization
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Scatter plot
        st.subheader("Feature Relationships")
        x_col = st.selectbox("Select X-axis feature", numeric_cols)
        y_col = st.selectbox("Select Y-axis feature", numeric_cols)
        
        fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        st.plotly_chart(fig)

    elif page == "Crop Prediction":
        st.title("🌾 Crop Recommendation System")
        st.write("Enter soil parameters and environmental conditions to get crop recommendations")

        # Create two columns for input
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Environmental Parameters")
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0, 0.1)
            rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 200.0, 1.0)
            ph = st.slider("pH value", 0.0, 14.0, 7.0, 0.1)

        with col2:
            st.subheader("Soil Parameters")
            nitrogen = st.slider("Nitrogen (N)", 0, 150, 50)
            phosphorus = st.slider("Phosphorus (P)", 0, 150, 50)
            potassium = st.slider("Potassium (K)", 0, 150, 50)

        if st.button("Get Crop Recommendation"):
            try:
                # Identify target column (assume it's the only non-numeric column)
                target_column = data.select_dtypes(include=['object']).columns[0]
                
                # Prepare the dataset
                X = data.drop(target_column, axis=1)
                y = data[target_column]

                # Encode target variable
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

                # Train the model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Prepare input data
                input_features = pd.DataFrame({
                    'N': [nitrogen],
                    'P': [phosphorus],
                    'K': [potassium],
                    'temperature': [temperature],
                    'humidity': [humidity],
                    'ph': [ph],
                    'rainfall': [rainfall]
                })

                # Make prediction
                prediction = model.predict(input_features)
                proba = model.predict_proba(input_features)
                
                # Get top 3 predictions
                top_3_idx = np.argsort(proba[0])[-3:][::-1]
                top_3_crops = label_encoder.inverse_transform(top_3_idx)
                top_3_probabilities = proba[0][top_3_idx]

                # Display results
                st.success(f"Top Recommended Crop: **{top_3_crops[0]}**")
                
                # Show confidence levels
                st.subheader("Top 3 Recommendations")
                for crop, prob in zip(top_3_crops, top_3_probabilities):
                    st.write(f"{crop}: {prob*100:.1f}% confidence")

                # Create confidence chart
                confidence_data = pd.DataFrame({
                    'Crop': top_3_crops,
                    'Confidence': top_3_probabilities * 100
                })
                fig = px.bar(confidence_data, x='Crop', y='Confidence',
                            title='Recommendation Confidence Levels')
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.write("Error details:", e)

    elif page == "Model Analysis":
        st.title("🎯 Model Performance Analysis")
        
        try:
            # Identify target column
            target_column = data.select_dtypes(include=['object']).columns[0]
            
            # Prepare data
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            # Encode target
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate metrics
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_score:.2%}")
            with col2:
                st.metric("Testing Accuracy", f"{test_score:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.subheader("Feature Importance")
            fig = px.bar(feature_importance, x='Feature', y='Importance',
                        title='Feature Importance in Crop Prediction')
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"An error occurred during model analysis: {str(e)}")
            st.write("Error details:", e)

else:
    st.error("Please ensure you have a valid dataset file named 'Crop_recommendation.csv' in the same directory as this script.")
    st.write("The dataset should contain the following columns:")
    st.write("- N (Nitrogen content)")
    st.write("- P (Phosphorus content)")
    st.write("- K (Potassium content)")
    st.write("- temperature")
    st.write("- humidity")
    st.write("- ph")
    st.write("- rainfall")
    st.write("- crop (target variable)")