import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from datetime import timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Walmart Sales Predictor", layout="wide")

st.title("🛒 Walmart Sales Forecasting System")
st.markdown("Comparing XGBoost vs. Random Forest for Retail Prediction")

# --- DATA LOADING ---
@st.cache_data
def load_real_data():
    try:
        # 1. Load the CSV
        df = pd.read_csv('Walmart.csv')
        
        # 2. Clean column names (removes hidden spaces like ' Weekly_Sales')
        df.columns = df.columns.str.strip()
        
        # 3. Fix Date Parsing (handles 19-02-2010 format)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # 4. Drop any rows where Date couldn't be parsed
        df = df.dropna(subset=['Date'])
        
        # 5. Extract Time Features
        df['Month'] = df['Date'].dt.month
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Year'] = df['Date'].dt.year
        
        # 6. Safety Check for IsHoliday column
        if 'IsHoliday' not in df.columns:
            df['IsHoliday'] = 0
        else:
            # Convert Boolean/String to 1/0
            df['IsHoliday'] = df['IsHoliday'].astype(int)
            
        return df
    except FileNotFoundError:
        st.error("🚨 Error: 'Walmart.csv' not found in this folder!")
        return None
    except Exception as e:
        st.error(f"🚨 An error occurred: {e}")
        return None

df = load_real_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Forecast Settings")
    
    # Store and Model selection
    available_stores = sorted(df['Store'].unique())
    store_id = st.sidebar.selectbox("Select Store ID", available_stores)
    model_choice = st.sidebar.radio("Select Predictive Model", ["XGBoost", "Random Forest"])
    forecast_weeks = st.sidebar.slider("Weeks to Forecast", 1, 12, 4)

    # --- MODELING ENGINE ---
    def generate_prediction(store_num, model_type):
        # Filter for selected store
        store_df = df[df['Store'] == store_num].sort_values('Date')
        
        # Features & Target
        # Note: We use Month, Week, and Holiday status to predict Sales
        features = ['Month', 'WeekOfYear', 'IsHoliday']
        X = store_df[features]
        y = store_df['Weekly_Sales']
        
        # Model Initialization
        if model_type == "XGBoost":
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        model.fit(X, y)
        
        # Create Future Dates for Forecasting
        last_date = store_df['Date'].max()
        future_dates = [last_date + timedelta(weeks=i) for i in range(1, forecast_weeks + 1)]
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Month': [d.month for d in future_dates],
            'WeekOfYear': [d.isocalendar().week for d in future_dates],
            'IsHoliday': [0] * len(future_dates) # Defaulting to 0 for forecast period
        })
        
        future_df['Predicted_Sales'] = model.predict(future_df[features])
        return store_df.tail(12), future_df

    # Run the prediction
    hist, fore = generate_prediction(store_id, model_choice)

    # --- UI DISPLAY ---
    col1, col2 = st.columns([3, 1])

    with col1:
        fig = go.Figure()
        # Historical Data
        fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Weekly_Sales'], 
                                 name="Historical Sales", line=dict(color='#1f77b4', width=3)))
        # Forecast Data
        fig.add_trace(go.Scatter(x=fore['Date'], y=fore['Predicted_Sales'], 
                                 name=f"{model_choice} Forecast", line=dict(color='#ff7f0e', width=4, dash='dash')))
        
        fig.update_layout(
            title=f"Sales Analysis: Store {store_id} ({model_choice})",
            xaxis_title="Date",
            yaxis_title="Weekly Sales ($)",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Model Selected", model_choice)
        st.metric("Total Projected Sales", f"${fore['Predicted_Sales'].sum():,.2f}")
        avg_sales = hist['Weekly_Sales'].mean()
        st.metric("Historical Avg", f"${avg_sales:,.2f}")
        
        st.write("---")
        st.success("Data loaded successfully!")

    # Optional: Show the math/logic
    with st.expander("Show Detailed Prediction Table"):
        st.dataframe(fore[['Date', 'Predicted_Sales']])

else:
    st.info("Check your folder to ensure 'Walmart.csv' is present and formatted correctly.")