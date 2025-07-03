import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models and scalers
classifier = joblib.load("Model/classifier_model.pkl")
classifier_scaler = joblib.load("Model/classifier_scaler.pkl")
regressor = joblib.load("Model/regressor_model.pkl")
regressor_scaler = joblib.load("Model/regressor_scaler.pkl")

# Encoding maps
yes_no_map = {"Yes": 1, "No": 0}
furnish_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}

# Required columns for prediction
required_columns = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "parking",
    "prefarea", "furnishingstatus"
]

# App UI
st.set_page_config(page_title="Real Estate Predictor", layout="wide")
st.title("🏡 Real Estate Bulk Prediction Tool + Visual Analysis")
st.write("Upload a dataset, predict profitability, and explore key trends visually.")

uploaded_file = st.file_uploader("📤 Upload your real estate CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load and filter
        df_raw = pd.read_csv(uploaded_file)
        missing = [col for col in required_columns if col not in df_raw.columns]
        if len(missing) > 0:
            st.warning(f"⚠️ Missing columns: {', '.join(missing)}")

        available_cols = [col for col in required_columns if col in df_raw.columns]
        df = df_raw[available_cols].copy()

        # Smart encode
        for col in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].map(yes_no_map)
        if "furnishingstatus" in df.columns and df["furnishingstatus"].dtype == object:
            df["furnishingstatus"] = df["furnishingstatus"].map(furnish_map)

        df_clean = df.dropna()

        if df_clean.empty:
            st.error("❌ No complete rows found with required data.")
        else:
            # Predict
            X_cls = classifier_scaler.transform(df_clean)
            profit_pred = classifier.predict(X_cls)
            profit_proba = classifier.predict_proba(X_cls)[:, 1]

            X_reg = regressor_scaler.transform(df_clean)
            time_pred = regressor.predict(X_reg)

            result_df = df_clean.copy()
            result_df["Profitable"] = np.where(profit_pred == 1, "Yes", "No")
            result_df["Confidence (%)"] = (profit_proba * 100).round(2)
            result_df["Time to Profit (Months)"] = time_pred.round(1)
            result_df["Time to Profit (Years)"] = (time_pred / 12).round(1)

            st.success("✅ Predictions complete!")
            st.dataframe(result_df)

            st.download_button(
                label="📥 Download Predictions CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="real_estate_predictions.csv",
                mime="text/csv"
            )

            # --- Visuals ---
            st.subheader("📊 Profitability Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=result_df, x="Profitable", palette="Set2", ax=ax1)
            ax1.set_title("Number of Profitable vs Not Profitable Properties")
            st.pyplot(fig1)

            st.subheader("⏳ Time to Profit Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(result_df["Time to Profit (Months)"], kde=True, bins=20, color="skyblue", ax=ax2)
            ax2.set_title("Distribution of Time to Profit (Months)")
            st.pyplot(fig2)

            with st.expander("📈 Correlation Plot (Optional)"):
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.heatmap(result_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
                ax3.set_title("Correlation Matrix")
                st.pyplot(fig3)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
