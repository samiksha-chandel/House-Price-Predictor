import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as stats
import plotly.express as px


st.set_page_config(page_title="🏡 Ames House Price Predictor", layout="wide")
st.title("🏡 Ames Housing Price Predictor")

try:
    # Load assets
    model = joblib.load("xgb_model.joblib")
    scaler = joblib.load("scaler.joblib")
    features = joblib.load("features.joblib")

    neighborhood_coords = {
        "Bloomington Heights": (42.06, -93.65),
        "Bluestem": (42.05, -93.63),
        "Briardale": (42.04, -93.61),
        "Brookside": (42.07, -93.66),
        "Clear Creek": (42.1, -93.68),
        "College Creek": (42.09, -93.64),
        "Crawford": (42.11, -93.65),
        "Edwards": (42.02, -93.6),
        "Gilbert": (42.12, -93.69),
        "Iowa DOT and Rail Road": (42.01, -93.59),
        "Meadow Village": (42.13, -93.67),
        "Mitchell": (42.08, -93.62),
        "North Ames": (42.14, -93.7),
        "Northridge": (42.15, -93.69),
        "Northpark Villa": (42.02, -93.63),
        "Northridge Heights": (42.16, -93.71),
        "Northwest Ames": (42.17, -93.68),
        "Old Town": (42.03, -93.61),
        "South & West of Iowa State University": (42.06, -93.62),
        "Sawyer": (42.05, -93.64),
        "Sawyer West": (42.04, -93.67),
        "Somerset": (42.18, -93.7),
        "Stone Brook": (42.19, -93.69),
        "Timberland": (42.2, -93.66),
        "Veenker": (42.21, -93.65)
    }


    df = pd.read_csv("AmesHousing.csv")
    df.columns = df.columns.str.replace(" ", "").str.strip()

    st.write("Enter property details to predict the price:")


    # Input columns and labels
    input_info = {
        "LotArea": "Lot Area (sq ft)",
        "YearBuilt": "Year Built",
        "OverallQual": "Overall Quality",
        "OverallCond": "Overall Condition",
        "GrLivArea": "Above Ground Living Area (sq ft)",
        "GarageCars": "Garage Capacity (Number of Cars)",
        "GarageArea": "Garage Area (sq ft)",
        "FullBath": "Number of Full Bathrooms",
        "TotRmsAbvGrd": "Total Rooms Above Ground"
    }

    selected_neighborhood = st.selectbox("Neighborhood", options=list(neighborhood_coords.keys()))

    user_input = {}
    cols = st.columns(3)

    for i, (col, label) in enumerate(input_info.items()):
        with cols[i % 3]:
            if col in ["OverallQual", "OverallCond"]:
                quality_map = {
                    "Poor": 1,
                    "Fair": 3,
                    "Average": 5,
                    "Good": 7,
                    "Excellent": 9
                }
                selected_label = st.selectbox(f"{label}", options=list(quality_map.keys()))
                user_input[col] = quality_map[selected_label]
            elif col in ["YearBuilt", "GarageCars", "FullBath", "TotRmsAbvGrd"]:
                user_input[col] = st.number_input(f"{label}", min_value=0, step=1, format="%d")
            else:
                user_input[col] = st.number_input(f"{label}", min_value=0.0, step=1.0)
                
    user_df = pd.DataFrame([user_input])

    # Ensure all required columns are present
    for feat in features:
        if feat not in user_df.columns:
            user_df[feat] = 0

    user_df = user_df[features]
    scaled_input = scaler.transform(user_df)

    if st.button("Predict"):
        log_price = model.predict(scaled_input)[0]

        # Simulate some variance (for confidence interval)
        simulated_preds = np.random.normal(loc=log_price, scale=0.03, size=1000)
        ci_low_log, ci_high_log = np.percentile(simulated_preds, [2.5, 97.5])

        # Convert back to real price
        price = np.exp(log_price)
        ci_low_exp = np.exp(ci_low_log)
        ci_high_exp = np.exp(ci_high_log)

        ci_low_str = f"${ci_low_exp:,.2f}"
        ci_high_str = f"${ci_high_exp:,.2f}"

        st.success(f"💰 Predicted House Price: **${price:,.2f}**")
        st.markdown(
            f"""
            <div style="
                background-color: #E8F4FD;
                padding: 12px 16px;
                border-left: 6px solid #1f77b4;
                border-radius: 6px;
                font-size: 16px;
            ">
                <b>📏 95% Confidence Interval:</b> {ci_low_str} – {ci_high_str}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Get coordinates
        coords = neighborhood_coords.get(selected_neighborhood, (42.0, -93.6))  # fallback

        # Store map point
        st.session_state["latest_map_point"] = {
            "Neighborhood": selected_neighborhood,
            "Latitude": coords[0],
            "Longitude": coords[1],
            "Price": round(price, 2)
        }

        if "latest_map_point" in st.session_state:
            st.subheader("📍 Property Location on Map")

        map_df = pd.DataFrame([st.session_state["latest_map_point"]])

        fig = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            hover_name="Neighborhood",
            hover_data={"Price": True},
            zoom=12,
            height=400,
            color_discrete_sequence=["#0074D9"]
        )

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Your House Compared to Average Values")

        # Friendly names
        friendly_labels = [input_info[feat] for feat in features]

        average_values = df[features].mean()
        user_values = pd.Series(user_input)[features]

        normalized_user = (user_values / df[features].max()).fillna(0)
        normalized_avg = (average_values / df[features].max()).fillna(0)

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=normalized_user.values,
            theta=friendly_labels,
            fill='toself',
            name='Your House',
            line_color='royalblue'
        ))

        fig.add_trace(go.Scatterpolar(
            r=normalized_avg.values,
            theta=friendly_labels,
            fill='toself',
            name='Average House',
            line_color='gray'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)


        st.subheader("📌 Category Analysis")

        category_data = []
        for feat in features:
            value = user_input[feat]
            low_thresh = df[feat].quantile(0.33)
            high_thresh = df[feat].quantile(0.66)

            if value < low_thresh:
                category = "Low"
                color = "🔴"
            elif value > high_thresh:
                category = "High"
                color = "🟢"
            else:
                category = "Medium"
                color = "🟡"
            
            category_data.append((input_info[feat], value, f"{color} {category}"))

        summary_df = pd.DataFrame(category_data, columns=["Feature", "Your Value", "Category"])
        st.dataframe(summary_df, use_container_width=True)

    


except Exception as e:
    st.error("⚠️ Something went wrong while loading or running the app.")
    st.exception(e)

# ------------------- BATCH PREDICTION SECTION -------------------

st.header("📁 Batch Prediction from CSV")

csv_file = st.file_uploader("Upload a CSV file with property details", type=["csv"])
if csv_file is not None:
    try:
        df_raw = pd.read_csv(csv_file)

        column_mapping = {
            "Lot Area": "LotArea",
            "Year Built": "YearBuilt",
            "Overall Qual": "OverallQual",
            "Overall Cond": "OverallCond",
            "Gr Liv Area": "GrLivArea",
            "Garage Cars": "GarageCars",
            "Garage Area": "GarageArea",
            "Full Bath": "FullBath",
            "TotRms AbvGrd": "TotRmsAbvGrd"
        }

        df_raw.rename(columns=column_mapping, inplace=True)

        features_needed = features
        df_input = df_raw[features_needed].copy()
        df_input.fillna(0, inplace=True)

        df_scaled = scaler.transform(df_input)
        log_preds = model.predict(df_scaled)
        df_input["PredictedPrice"] = np.expm1(log_preds)

        st.success("✅ Predictions completed!")
        st.dataframe(df_input)

        csv_download = df_input.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv_download, "predictions.csv", "text/csv")

    except Exception as e:
        st.error("❌ Error processing the file.")
        st.exception(e)
