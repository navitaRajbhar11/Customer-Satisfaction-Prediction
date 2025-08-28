import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ZenML / Pipeline imports
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main as run_pipeline_main


def load_images():
    """Load and cache images."""
    high_level = Image.open("_assets/high_level_overview.png")
    pipeline_overview = Image.open("_assets/training_and_deployment_pipeline_updated.png")
    feature_importance = Image.open("_assets/feature_importance_gain.png")
    return high_level, pipeline_overview, feature_importance


def predict_customer_satisfaction(service, features: dict):
    """Prepare input features and return prediction."""
    df = pd.DataFrame([features])
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    return service.predict(data)


def main():
    st.set_page_config(
        page_title="Customer Satisfaction Prediction",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 End-to-End Customer Satisfaction Pipeline with ZenML")

    # Load static assets
    high_level_image, whole_pipeline_image, feature_importance_image = load_images()

    st.image(high_level_image, caption="High Level Pipeline", use_column_width=True)

    st.markdown(
        """
        ### 📝 Problem Statement  
        The objective is to predict the **customer satisfaction score** for a given order 
        based on features like order status, price, payment, and product characteristics.  

        This app uses **[ZenML](https://zenml.io/)** pipelines to build a 
        production-ready ML system for prediction.  
        """
    )

    st.image(whole_pipeline_image, caption="Training & Deployment Pipeline", use_column_width=True)

    st.markdown(
        """
        The pipeline consists of:  
        - Data ingestion  
        - Data cleaning  
        - Model training and evaluation  
        - Continuous deployment with MLflow  
        """
    )

    st.divider()

    # --- Sidebar Inputs ---
    st.sidebar.header("🔧 Input Features")

    features = {
        "payment_sequential": st.sidebar.slider("Payment Sequential", 0, 10, 1),
        "payment_installments": st.sidebar.slider("Payment Installments", 0, 24, 1),
        "payment_value": st.sidebar.number_input("Payment Value", min_value=0.0, value=100.0),
        "price": st.sidebar.number_input("Price", min_value=0.0, value=50.0),
        "freight_value": st.sidebar.number_input("Freight Value", min_value=0.0, value=10.0),
        "product_name_lenght": st.sidebar.number_input("Product Name Length", min_value=0, value=10),
        "product_description_lenght": st.sidebar.number_input("Product Description Length", min_value=0, value=50),
        "product_photos_qty": st.sidebar.number_input("Product Photos Quantity", min_value=0, value=1),
        "product_weight_g": st.sidebar.number_input("Product Weight (grams)", min_value=0, value=500),
        "product_length_cm": st.sidebar.number_input("Product Length (cm)", min_value=0, value=20),
        "product_height_cm": st.sidebar.number_input("Product Height (cm)", min_value=0, value=10),
        "product_width_cm": st.sidebar.number_input("Product Width (cm)", min_value=0, value=15),
    }

    # --- Prediction Button ---
    if st.button("🚀 Predict Satisfaction Score"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )

        if service is None:
            st.warning("⚠️ No active service found. Running pipeline to deploy model...")
            run_pipeline_main()
            service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                running=True,
            )

        prediction = predict_customer_satisfaction(service, features)
        st.success(f"✅ Predicted Customer Satisfaction Score: **{prediction}** (range 0-5)")

    # --- Results / Model Comparison ---
    if st.button("📊 Show Results"):
        st.subheader("Model Performance Comparison")

        results_df = pd.DataFrame(
            {
                "Models": ["LightGBM", "XGBoost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(results_df)

        st.markdown("### 🔎 Feature Importance")
        st.image(feature_importance_image, caption="Feature Importance (Gain)", use_column_width=True)


if __name__ == "__main__":
    main()
