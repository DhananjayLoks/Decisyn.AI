import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
import io
import os
import google.generativeai as genai
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURE API (Required for Localhost) ---
# Paste your API key here
genai.configure(api_key="GOOGLE_API_KEY")

# --- 2. PASTE THE HELPER FUNCTION HERE ---
# It must be here so Python reads it FIRST
def generate_excel_from_prompt(prompt_text):
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Define the prompt strictly
        system_instruction = f"""
        You are a python data generator. 
        User Request: "{prompt_text}"
        
        RULES:
        1. Write a Python script using pandas and numpy.
        2. Create a DataFrame containing the requested data.
        3. YOU MUST assign the final dataframe to a variable named 'df_generated'.
        4. Do not use print(). Do not output markdown text. Just the code.
        """
        
        # Get response
        response = model.generate_content(system_instruction)
        logic_code = response.text 
        
        # Clean code
        logic_code = logic_code.replace("```python", "").replace("```", "")
        
        # Execute
        local_vars = {}
        exec(logic_code, globals(), local_vars)

        if 'df_generated' not in local_vars:
            return None, "AI failed to generate data."
        
        df = local_vars['df_generated']

        # Save to memory buffer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)
        
        return output, "Success"

    except Exception as e:
        return None, f"Error: {str(e)}"

# --- ML Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# --- Optional Libraries (Safety Switches) ---
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Decisyn AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🧊 V4.2 CSS ENGINE (Rectangular Cards & Colors)
# ==========================================
st.markdown("""
    <style>
    /* GLOBAL THEME */
    .stApp {
        background-color: #e0e5ec; /* Neumorphic grey background */
        font-family: 'Segoe UI', sans-serif;
    }

    /* 1. JIGGLY KPI CARDS (Rectangular & 3D) */
    @keyframes jiggle {
        0% { transform: rotate(-1deg); }
        50% { transform: rotate(1deg); }
        100% { transform: rotate(-1deg); }
    }
    
    .kpi-card {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        border-radius: 15px;
        padding: 15px 20px; /* Wider padding for rectangular feel */
        margin-bottom: 20px;
        /* Neumorphic 3D Shadow */
        box-shadow: 6px 6px 12px rgb(163,177,198,0.5), -6px -6px 12px rgba(255,255,255, 0.8);
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
        text-align: center;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 140px; /* Fixed height to ensure rectangular look */
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        animation: jiggle 0.4s ease-in-out;
        box-shadow: 10px 10px 20px rgb(163,177,198,0.6), -10px -10px 20px rgba(255,255,255, 0.9);
    }
    
    /* Title Top */
    .kpi-title { 
        color: #7f8c8d; 
        font-size: 13px; 
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 1.2px; 
        margin-bottom: 5px;
    }
    
    /* Value Middle (Large) */
    .kpi-value { 
        color: #2c3e50; 
        font-size: 32px; 
        font-weight: 900; 
        margin: 5px 0 15px 0; 
    }
    
    /* Footer Bottom (Horizontal) */
    .kpi-footer { 
        display: flex;
        justify-content: space-around;
        align-items: center;
        font-size: 11px; 
        font-weight: 600;
        background: rgba(0,0,0,0.03); 
        padding: 8px; 
        border-radius: 8px; 
        width: 100%;
    }

    /* 2. CHART CONTAINERS (Glassmorphism) */
    .chart-container {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }

    /* 3. CLEAN UP STREAMLIT UI */
    [data-testid="stSidebar"] { background-color: #e0e5ec; border-right: 1px solid #fff; }
    .stPopover { border: none; }
    
    /* Preserved Styles */
    .metric-card {
        background-color: #fff; border-radius: 15px; padding: 15px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)


# --- Gemini AI Configuration ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY")

    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        GEMINI_CONFIGURED = True
    else:
        GEMINI_CONFIGURED = False
except Exception as e:
    GEMINI_CONFIGURED = False

# --- Caching Functions ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = None
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def add_to_history(df):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(df.copy())
    if len(st.session_state.history) > 5:
        st.session_state.history.pop(0)

# ==========================================
# 🚀 V4.2 DASHBOARD ENGINE (CUSTOMIZED & PRESERVED)
# ==========================================
def render_ocd_dashboard(df):
    st.markdown("## 🧊 3D Corporate Command Center")

    # 1. Run Universal Intelligence
    nums = df.select_dtypes(include=['number']).columns.tolist()
    dates = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Try to find a date column even if it's currently object/text
    if not dates:
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head(), errors='raise')
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    dates.append(col)
                except:
                    pass
    
    dt_col = dates[0] if dates else None
    
    # Smart Category Filters
    cats = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() < 20]
    high_cats = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() > 10]

    # --- ROW 1: CUSTOM RECTANGULAR KPI CARDS ---
    st.markdown("### 🎯 Executive Summary")
    
    selected_kpis = st.sidebar.multiselect("Select KPIs (Top Row)", nums, default=nums[:4] if len(nums)>=4 else nums)
    
    if selected_kpis:
        cols = st.columns(len(selected_kpis))
        # Accent colors for the top border only
        colors = ["#4c7cff", "#00cc96", "#ffb020", "#ff4b4b", "#9c27b0"] 
        
        for i, col in enumerate(selected_kpis):
            val = df[col].sum()
            avg = df[col].mean()
            mx = df[col].max()
            mn = df[col].min()
            color = colors[i % len(colors)]
            
            with cols[i]:
                # CUSTOM HTML: Title Top, Big Number Middle, Colored Footer
                st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid {color};">
                    <div class="kpi-title">{col}</div>
                    <div class="kpi-value">{val:,.0f}</div>
                    <div class="kpi-footer">
                        <span style="color: #ff4b4b;">▼ Low: {mn:,.0f}</span>
                        <span style="color: #f1c40f;">〰 Avg: {avg:,.0f}</span>
                        <span style="color: #00cc96;">▲ High: {mx:,.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- ROW 2: STRATEGIC (2:1 Ratio) ---
    c_mid_1, c_mid_2 = st.columns([2, 1])
    
    # 1. FUNNEL (Left)
    with c_mid_1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        h1, h2 = st.columns([8, 1])
        with h1: st.markdown("#### 🌪️ Process Funnel")
        
        with h2:
            with st.popover("⚙️"):
                st.markdown("**Chart Settings**")
                f_cat = st.selectbox("Funnel Stage:", cats, index=0 if cats else None)
                f_val = st.selectbox("Funnel Value:", nums, index=0 if nums else None)
        
        if f_cat and f_val:
            df_fun = df.groupby(f_cat)[f_val].sum().reset_index().sort_values(f_val, ascending=False)
            fig = px.funnel(df_fun, x=f_val, y=f_cat, color=f_val) 
            fig.update_layout(showlegend=False, height=350, margin=dict(t=0,b=0,l=0,r=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="funnel_3d")
        else:
            st.info("Need categorical data for Funnel.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. ACCELERATOR / SPEEDOMETER (Right) - WITH EDIT BUTTON
    with c_mid_2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        h1, h2 = st.columns([7, 2])
        with h1: st.markdown("#### 🏎️ Velocity")
        
        # --- FIX: EDIT BUTTON ADDED TO VELOCITY ---
        default_gauge_idx = 0
        if nums and 'Revenue' in nums: default_gauge_idx = nums.index('Revenue')
        elif nums and 'Unit_Price' in nums: default_gauge_idx = nums.index('Unit_Price')

        with h2:
            with st.popover("⚙️"):
                gauge_metric = st.selectbox("Measure:", nums, index=default_gauge_idx if nums else 0)
        
        if gauge_metric:
            curr_avg = df[gauge_metric].mean()
            max_val = df[gauge_metric].max()
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = curr_avg,
                title = {'text': f"Avg {gauge_metric}", 'font': {'size': 14}},
                gauge = {
                    'axis': {'range': [0, max_val]},
                    'bar': {'color': "#00cc96"}, # Neon Green
                    'steps': [
                        {'range': [0, max_val*0.3], 'color': "rgba(255, 75, 75, 0.2)"},
                        {'range': [max_val*0.3, max_val*0.7], 'color': "rgba(255, 255, 0, 0.2)"},
                        {'range': [max_val*0.7, max_val], 'color': "rgba(0, 255, 0, 0.2)"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(t=30,b=10,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "black"})
            st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_3d")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- ROW 3: BREAKDOWN (1:2 Ratio) ---
    c_low_1, c_low_2 = st.columns([1, 2])
    
    # 1. PIE/DONUT (Left)
    with c_low_1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        h1, h2 = st.columns([8, 1])
        with h1: st.markdown("#### 🍩 Distribution")
        
        default_pie_idx = 1 if len(cats) > 1 else 0
        
        with h2:
            with st.popover("⚙️"):
                p_cat = st.selectbox("Slice By:", cats, index=default_pie_idx if cats else None, key="pie_cat")
                p_val = st.selectbox("Size By:", nums, index=0 if nums else None, key="pie_val")
        
        if p_cat and p_val:
            fig_pie = px.pie(df, names=p_cat, values=p_val, hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=False, height=300, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_3d")
        else:
            st.info("No categories for Pie Chart.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. BAR CHART (Right) - WITH AUTO-AGGREGATION
    with c_low_2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        h1, h2 = st.columns([9, 1])
        
        is_trend = True if dt_col else False
        chart_title = "📅 Time Analysis" if is_trend else "📊 Categorical Analysis"
        
        with h1: st.markdown(f"#### {chart_title}")
        
        with h2:
            with st.popover("⚙️"):
                b_x = st.selectbox("X Axis:", dates if dates else cats, index=0, key="bar_x")
                b_y = st.selectbox("Y Axis:", nums, index=0 if nums else None, key="bar_y")
                b_color = st.selectbox("Color By:", cats if cats else None, index=0 if cats else None, key="bar_color")

        if b_x and b_y:
            # === FIX: AUTO-SUMMARIZE DATA ===
            plot_df = df.copy()
            # If X is a date and we have > 30 points, group by Month
            if b_x in dates and df[b_x].nunique() > 30:
                plot_df[b_x] = plot_df[b_x].dt.to_period('M').astype(str)
                # Group grouping
                grp_cols = [b_x]
                if b_color: grp_cols.append(b_color)
                plot_df = plot_df.groupby(grp_cols)[b_y].sum().reset_index()
                st.caption(f"ℹ️ Auto-grouped by Month for visibility (Too many daily points).")
            
            fig_bar = px.bar(plot_df, x=b_x, y=b_y, color=b_color if b_color else None,
                             template="plotly_white", barmode='group')
            fig_bar.update_layout(height=300, margin=dict(t=10,b=0,l=0,r=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_smart")
        else:
            st.info("Not enough data for Bar Chart.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- ROW 4: HAPPY ENDING (50/50) ---
    c_end_1, c_end_2 = st.columns(2)
    
    # 1. LEADERBOARD (Top 5)
    with c_end_1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        h1, h2 = st.columns([8, 1])
        with h1: st.markdown("#### 🏆 Champions (Top 5)")
        
        with h2:
            with st.popover("⚙️"):
                l_cat = st.selectbox("Entity:", high_cats if high_cats else cats, index=0, key="lead_cat")
                l_val = st.selectbox("Metric:", nums, index=0, key="lead_val")
        
        if l_cat and l_val:
            df_top = df.groupby(l_cat)[l_val].sum().reset_index().sort_values(l_val, ascending=False).head(5)
            st.dataframe(df_top, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. PERFORMANCE HORIZONTAL BAR (Performance)
    with c_end_2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        h1, h2 = st.columns([8, 1])
        with h1: st.markdown("#### ⚡ Performance Ranking")
        
        with h2:
            with st.popover("⚙️"):
                perf_cat = st.selectbox("Item:", high_cats if high_cats else cats, index=0, key="perf_cat")
                perf_val = st.selectbox("Value:", nums, index=0, key="perf_val")

        if perf_cat and perf_val:
            df_perf = df.groupby(perf_cat)[perf_val].sum().reset_index().sort_values(perf_val, ascending=True).tail(10) 
            fig_perf = px.bar(df_perf, x=perf_val, y=perf_cat, orientation='h',
                              color=perf_val, color_continuous_scale="Viridis")
            fig_perf.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_perf, use_container_width=True, key="perf_bar")
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 🧠 PRO ML STUDIO V2.0 (REBUILT & UPGRADED)
# ==========================================

def clean_and_encode_ml(df, target_col):
    df = df.copy()
    # 1. Fill Missing Values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # 2. Encode Categorical
    encoders = {}
    for col in df.columns:
        if col == target_col: continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def render_ml_studio(df):
    st.markdown("## 🧠 Pro ML Studio")
    
    # Initialize Session State
    if 'trained_model' not in st.session_state: st.session_state.trained_model = None
    if 'model_features' not in st.session_state: st.session_state.model_features = None
    if 'model_scaler' not in st.session_state: st.session_state.model_scaler = None
    if 'model_task' not in st.session_state: st.session_state.model_task = None

    tab_build, tab_sim, tab_unsup = st.tabs(["🏗️ Model Builder", "🔮 What-If Simulator", "🌌 Unsupervised Lab"])

    # --- TAB 1: MODEL BUILDER ---
    with tab_build:
        c1, c2 = st.columns([1, 2], gap="large")
        
        with c1:
            st.markdown("### 1. Configuration")
            target_col = st.selectbox("🎯 Target Variable (Y)", df.columns)
            
            # Auto-Detect Task
            is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
            task = "Regression" if is_numeric else "Classification"
            st.info(f"Detected Task: **{task}**")
            
            # Feature Selection
            avail_cols = [c for c in df.columns if c != target_col]
            features = st.multiselect("📊 Independent Variables (X)", avail_cols, default=avail_cols[:4] if len(avail_cols) > 4 else avail_cols)
            
            # Algorithm Selection
            st.markdown("### 2. Algorithm")
            mode = st.radio("Mode", ["Auto-Compare (Best Practice)", "Manual Selection"])
            
            selected_model = None
            params = {}
            
            if mode == "Manual Selection":
                if task == "Regression":
                    algos = ["Linear Regression", "Random Forest", "Decision Tree", "Support Vector Machine"]
                    if HAS_XGB: algos.append("XGBoost")
                else:
                    algos = ["Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine", "K-Neighbors"]
                    if HAS_XGB: algos.append("XGBoost")
                
                selected_model = st.selectbox("Choose Model", algos)
                
                # Dynamic Hyperparameters
                if selected_model in ["Random Forest", "XGBoost"]:
                    params['n_estimators'] = st.slider("Number of Trees", 10, 500, 100)
                    params['max_depth'] = st.slider("Max Depth", 1, 20, 10)
                elif selected_model == "Decision Tree":
                    params['max_depth'] = st.slider("Max Depth", 1, 20, 5)
            
            split_size = st.slider("Train/Test Split", 0.1, 0.5, 0.2)
            train_btn = st.button("🚀 Train Model(s)", type="primary", use_container_width=True)

        with c2:
            st.markdown("### 3. Results Arena")
            if train_btn:
                if not features:
                    st.error("⚠️ Please select at least one independent variable (X).")
                else:
                    with st.spinner("Training & Evaluating..."):
                        try:
                            # Data Prep
                            df_train = df[features + [target_col]].copy()
                            df_train, encoders = clean_and_encode_ml(df_train, target_col)
                            X = df_train[features]
                            y = df_train[target_col]
                            
                            if task == "Classification" and y.dtype == 'object':
                                le_target = LabelEncoder()
                                y = le_target.fit_transform(y)
                                
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                            
                            # Define Models
                            models = {}
                            if task == "Regression":
                                models["Linear Regression"] = LinearRegression()
                                models["Random Forest"] = RandomForestRegressor(n_estimators=params.get('n_estimators', 100))
                                models["Decision Tree"] = DecisionTreeRegressor(max_depth=params.get('max_depth', None))
                                models["Support Vector Machine"] = SVR()
                                if HAS_XGB: models["XGBoost"] = XGBRegressor(n_estimators=params.get('n_estimators', 100))
                            else:
                                models["Logistic Regression"] = LogisticRegression(max_iter=1000)
                                models["Random Forest"] = RandomForestClassifier(n_estimators=params.get('n_estimators', 100))
                                models["Decision Tree"] = DecisionTreeClassifier(max_depth=params.get('max_depth', None))
                                models["Support Vector Machine"] = SVC(probability=True)
                                models["K-Neighbors"] = KNeighborsClassifier()
                                if HAS_XGB: models["XGBoost"] = XGBClassifier(n_estimators=params.get('n_estimators', 100))

                            results = []
                            
                            # Training Loop
                            models_to_run = models.items() if mode.startswith("Auto") else [(selected_model, models[selected_model])]
                            
                            best_score = -999
                            best_model_obj = None
                            best_model_name = ""

                            for name, model_obj in models_to_run:
                                model_obj.fit(X_train, y_train)
                                preds = model_obj.predict(X_test)
                                
                                if task == "Regression":
                                    score = r2_score(y_test, preds)
                                    mse = mean_squared_error(y_test, preds)
                                    results.append({"Model": name, "R2 Score": score, "MSE": mse})
                                else:
                                    score = accuracy_score(y_test, preds)
                                    results.append({"Model": name, "Accuracy": score})
                                
                                if score > best_score:
                                    best_score = score
                                    best_model_obj = model_obj
                                    best_model_name = name

                            # Save Best Model to State
                            st.session_state.trained_model = best_model_obj
                            st.session_state.model_features = features
                            st.session_state.model_scaler = scaler
                            st.session_state.model_task = task
                            
                            # Display Results
                            res_df = pd.DataFrame(results).sort_values(by="R2 Score" if task == "Regression" else "Accuracy", ascending=False)
                            
                            # 1. Leaderboard
                            st.success(f"🏆 Winner: **{best_model_name}** with Score: {best_score:.4f}")
                            st.dataframe(res_df, use_container_width=True, hide_index=True)
                            
                            # 2. Visualizations (UPDATED)
                            
                            # A) Feature Importance (Best for Trees/Forests)
                            if hasattr(best_model_obj, 'feature_importances_'):
                                st.markdown("#### 🔑 Feature Importance (The 'Why')")
                                imp_df = pd.DataFrame({
                                    'Feature': features,
                                    'Importance': best_model_obj.feature_importances_
                                }).sort_values('Importance', ascending=True)
                                
                                fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
                                st.plotly_chart(fig_imp, use_container_width=True)

                            # B) Regression Scatter Plot (Best for Linear/SVR/Regression Tasks)
                            # This replaces the old "Coefficients" chart for regression
                            elif task == "Regression":
                                st.markdown("#### 📉 Actual vs. Predicted (Scatter)")
                                
                                # Generate fresh predictions using the winner model
                                final_pred = best_model_obj.predict(X_test)
                                
                                # Create Matplotlib Figure
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Plot Data Points (Blue Cloud)
                                sns.scatterplot(x=y_test, y=final_pred, alpha=0.6, ax=ax, label="Data Points")
                                
                                # Plot Perfect Fit Line (Red Dashed)
                                mn = min(min(y_test), min(final_pred))
                                mx = max(max(y_test), max(final_pred))
                                ax.plot([mn, mx], [mn, mx], color='red', linestyle='--', linewidth=2, label="Perfect Fit")
                                
                                ax.set_xlabel("Actual Values")
                                ax.set_ylabel("Predicted Values")
                                ax.set_title(f"Model Performance: {best_model_name}")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)

                            # C) Fallback for Classification Linear Models (Logistic Regression)
                            elif hasattr(best_model_obj, 'coef_'):
                                st.markdown("#### 🔑 Coefficients (Impact)")
                                coefs = best_model_obj.coef_
                                if len(coefs.shape) > 1: coefs = coefs[0]
                                imp_df = pd.DataFrame({
                                    'Feature': features,
                                    'Weight': coefs
                                }).sort_values('Weight', ascending=True)
                                fig_imp = px.bar(imp_df, x='Weight', y='Feature', orientation='h')
                                st.plotly_chart(fig_imp, use_container_width=True)

                        except Exception as e:
                            st.error(f"Training Failed: {str(e)}")
    # --- TAB 2: WHAT-IF SIMULATOR ---
    with tab_sim:
        if st.session_state.trained_model is None:
            st.warning("👈 Please train a model in the 'Model Builder' tab first.")
        else:
            st.markdown(f"### 🔮 Simulator (Using {type(st.session_state.trained_model).__name__})")
            st.caption("Change values below to predict the outcome in real-time.")
            
            with st.form("sim_form"):
                inputs = {}
                cols = st.columns(3)
                for i, col in enumerate(st.session_state.model_features):
                    with cols[i % 3]:
                        # Intelligent Defaults: Use mean of data
                        default_val = float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                        inputs[col] = st.number_input(f"{col}", value=default_val)
                
                predict_btn = st.form_submit_button("Predict Outcome")
            
            if predict_btn:
                input_df = pd.DataFrame([inputs])
                # Scale
                input_scaled = st.session_state.model_scaler.transform(input_df)
                # Predict
                pred = st.session_state.trained_model.predict(input_scaled)[0]
                
                st.markdown("---")
                if st.session_state.model_task == "Regression":
                    st.metric("Predicted Value", f"{pred:,.2f}")
                else:
                    st.metric("Predicted Class", f"{pred}")

    # --- TAB 3: UNSUPERVISED LAB ---
    with tab_unsup:
        st.markdown("### 🌌 Pattern Discovery Lab")
        
        # Method Selector
        us_algo = st.selectbox("Select Method", ["K-Means Clustering (3D)", "Hierarchical Clustering (Dendrogram)", "PCA (Dimensionality Reduction)", "Apriori (Rules)", "Isolation Forest (Anomalies)"])
        
        if us_algo == "K-Means Clustering (3D)":
            c1, c2 = st.columns([1, 3])
            with c1:
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                k_feats = st.multiselect("Select 3 Features", num_cols, default=num_cols[:3] if len(num_cols)>=3 else num_cols)
                k_clusters = st.slider("Clusters (K)", 2, 8, 3)
                run_k = st.button("Run 3D K-Means")
                
            with c2:
                if run_k and len(k_feats) >= 3:
                    X = df[k_feats].dropna()
                    scaler = StandardScaler()
                    X_sc = scaler.fit_transform(X)
                    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
                    labels = kmeans.fit_predict(X_sc)
                    X['Cluster'] = labels.astype(str)
                    
                    # 3D Plot
                    fig_3d = px.scatter_3d(X, x=k_feats[0], y=k_feats[1], z=k_feats[2], color='Cluster',
                                           title="3D Customer Segmentation", opacity=0.8)
                    st.plotly_chart(fig_3d, use_container_width=True)
                elif run_k:
                    st.error("Please select at least 3 numeric features for 3D visualization.")

        elif us_algo == "Hierarchical Clustering (Dendrogram)":
            st.markdown("#### 🌳 Dendrogram Analysis")
            h_cols = st.multiselect("Features for Hierarchy", df.select_dtypes(include=np.number).columns, default=df.select_dtypes(include=np.number).columns[:5])
            threshold = st.slider("Cut Threshold", 0.0, 50.0, 7.0)
            
            if st.button("Generate Dendrogram"):
                if len(h_cols) > 0:
                    # Limit rows for Dendrogram performance
                    X_h = df[h_cols].dropna().sample(min(len(df), 200), random_state=42)
                    X_sc = StandardScaler().fit_transform(X_h)
                    
                    plt.figure(figsize=(10, 5))
                    plt.title("Hierarchical Clustering Dendrogram")
                    dend = shc.dendrogram(shc.linkage(X_sc, method='ward'))
                    plt.axhline(y=threshold, color='r', linestyle='--')
                    plt.text(0, threshold + 0.1, 'Threshold Line', color='r')
                    st.pyplot(plt)
                    st.info("ℹ️ The threshold line determines the number of clusters. Vertical lines crossing the red line represent distinct groups.")

        elif us_algo == "Apriori (Rules)":
            if HAS_MLXTEND:
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                b_col = st.selectbox("Transaction ID", df.columns)
                i_col = st.selectbox("Item Column", cat_cols)
                if st.button("Mine Rules"):
                    try:
                        basket = (df.groupby([b_col, i_col])[i_col].count().unstack().reset_index().fillna(0).set_index(b_col))
                        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
                        frq_items = apriori(basket, min_support=0.05, use_colnames=True)
                        rules = association_rules(frq_items, metric="lift", min_threshold=1)
                        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
                    except Exception as e:
                        st.error(f"Apriori Error: {e}")
            else:
                st.error("Library 'mlxtend' not installed. `pip install mlxtend` to use Apriori.")

        elif us_algo == "PCA (Dimensionality Reduction)":
            p_cols = st.multiselect("Features to Compress", df.select_dtypes(include=np.number).columns)
            if st.button("Run PCA"):
                if len(p_cols) > 1:
                    X_p = df[p_cols].dropna()
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(StandardScaler().fit_transform(X_p))
                    fig_pca = px.scatter(x=components[:,0], y=components[:,1], title="PCA Projection (2D)", labels={'x':'PC1', 'y':'PC2'})
                    st.plotly_chart(fig_pca, use_container_width=True)

        elif us_algo == "Isolation Forest (Anomalies)":
            a_cols = st.multiselect("Features for Anomaly Detection", df.select_dtypes(include=np.number).columns)
            contam = st.slider("Contamination (Expected % Outliers)", 0.01, 0.2, 0.05)
            if st.button("Detect Anomalies"):
                if len(a_cols) > 0:
                    X_a = df[a_cols].dropna()
                    iso = IsolationForest(contamination=contam, random_state=42)
                    preds = iso.fit_predict(StandardScaler().fit_transform(X_a))
                    X_a['Anomaly'] = preds
                    X_a['Anomaly'] = X_a['Anomaly'].map({1: 'Normal', -1: 'Outlier'})
                    
                    fig_iso = px.scatter(X_a, x=a_cols[0], y=a_cols[1], color='Anomaly', 
                                         color_discrete_map={'Normal': 'blue', 'Outlier': 'red'}, title="Anomaly Detection")
                    st.plotly_chart(fig_iso, use_container_width=True)

# --- UI FUNCTION FOR DATA ANALYZER ---
def page_data_analyzer():
    st.header("Analyze, Clean & Visualize Your Data")
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            st.error("Unsupported file format.")
            return

        if 'df_processed' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            st.session_state.df_processed = df.copy()
            st.session_state.file_name = uploaded_file.name
            st.session_state.history = [df.copy()]
        
        df_processed = st.session_state.df_processed
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizer", "🛠️ Data Explorer & Cleaner", "🧊 3D Dashboard", "🤖 Pro ML Studio"])
        
        # === TAB 1: VISUALIZER (PRESERVED) ===
        with tab1:
            st.subheader("Interactive Visualization")
            col1, col2 = st.columns([1, 2])
            with col1:
                chart_type = st.selectbox("Chart Type:", ["Bar", "Scatter", "Line", "Histogram", "Box Plot", "Pie Chart", "Bubble Chart", "Pareto Chart"])
                
                if chart_type == "Pareto Chart":
                    category_col = st.selectbox("Categorical column:", df_processed.select_dtypes(include=['object', 'category']).columns)
                    value_col = st.selectbox("Numerical column for aggregation:", df_processed.select_dtypes(include=['number']).columns)
                elif chart_type == "Bubble Chart":
                    x_axis = st.selectbox("X-axis:", df_processed.columns)
                    y_axis = st.selectbox("Y-axis:", df_processed.columns, index=1 if len(df_processed.columns) > 1 else 0)
                    size_col = st.selectbox("Column for bubble size:", df_processed.select_dtypes(include=['number']).columns)
                elif chart_type == "Pie Chart":
                    names_col = st.selectbox("Column for labels:", df_processed.columns)
                    values_col = st.selectbox("Column for values:", df_processed.columns, index=1 if len(df_processed.columns) > 1 else 0)
                elif chart_type == "Histogram":
                    x_axis = st.selectbox("Column for distribution:", df_processed.columns)
                else:
                    x_axis = st.selectbox("X-axis:", df_processed.columns, index=0)
                    y_axis = st.selectbox("Y-axis:", df_processed.columns, index=1 if len(df_processed.columns) > 1 else 0)

                chart_title = st.text_input("Chart Title", value=f"Chart for {uploaded_file.name}")
            
            with col2:
                try:
                    if chart_type == "Pareto Chart":
                        pareto_df = df_processed.groupby(category_col)[value_col].sum().reset_index().sort_values(by=value_col, ascending=False)
                        pareto_df['Cumulative Percentage'] = (pareto_df[value_col].cumsum() / pareto_df[value_col].sum()) * 100
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(go.Bar(x=pareto_df[category_col], y=pareto_df[value_col], name=value_col), secondary_y=False)
                        fig.add_trace(go.Scatter(x=pareto_df[category_col], y=pareto_df['Cumulative Percentage'], name='Cumulative Percentage', mode='lines+markers'), secondary_y=True)
                        fig.update_layout(title_text=chart_title)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        plot_args = {'data_frame': df_processed, 'title': chart_title}
                        if chart_type == "Pie Chart": fig = px.pie(df_processed, names=names_col, values=values_col, title=chart_title)
                        elif chart_type == "Histogram": fig = px.histogram(df_processed, x=x_axis, title=chart_title)
                        elif chart_type == "Bubble Chart": fig = px.scatter(df_processed, x=x_axis, y=y_axis, size=size_col, title=chart_title)
                        else:
                            plot_args.update({'x': x_axis, 'y': y_axis})
                            if chart_type == "Bar": fig = px.bar(**plot_args)
                            elif chart_type == "Scatter": fig = px.scatter(**plot_args)
                            elif chart_type == "Line": fig = px.line(**plot_args)
                            elif chart_type == "Box Plot": fig = px.box(**plot_args)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate chart: {e}")

        # === TAB 2: EXPLORER & CLEANER (PRESERVED) ===
        with tab2:
            st.subheader("🛠️ Advanced Data Cleaning Toolkit")
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            missing_cells = df_processed.isnull().sum().sum()
            total_cells = df_processed.size
            duplicate_rows = df_processed.duplicated().sum()
            
            with col_kpi1: st.metric("Data Completeness", f"{(1 - missing_cells/total_cells):.1%}")
            with col_kpi2: st.metric("Missing Values", missing_cells, delta=f"-{missing_cells}" if missing_cells > 0 else None, delta_color="inverse")
            with col_kpi3: st.metric("Duplicate Rows", duplicate_rows, delta=f"-{duplicate_rows}" if duplicate_rows > 0 else None, delta_color="inverse")
            
            st.markdown("---")
            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.caption("🤖 AI & Structure")
                
                with st.expander("🪄 Natural Language Cleaning (Experimental)"):
                    st.warning("⚠️ This feature is experimental. Always review the changes to your data carefully.")
                    nl_command = st.text_input("Enter a cleaning command in plain English:", placeholder="e.g., make the 'country' column lowercase")
                    if st.button("Execute Command"):
                        with st.spinner("🤖 Translating and executing..."):
                            try:
                                prompt = f"""You are a Pandas expert. A user wants to modify their DataFrame, named df_processed. Their command is: '{nl_command}'. Generate ONLY the single line of Python code to perform this and reassign it back to df_processed. Example: for 'make country uppercase', output df_processed['country'] = df_processed['country'].str.upper()."""
                                response = model.generate_content(prompt)
                                generated_code = response.text.strip().replace("python", "").replace("```", "")
                                
                                add_to_history(df_processed)
                                local_vars = {'df_processed': st.session_state.df_processed.copy(), 'pd': pd, 'np': np}
                                exec(generated_code, {}, local_vars)
                                st.session_state.df_processed = local_vars['df_processed']
                                st.success("✅ Command executed successfully.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not execute command. Error: {e}")

                with st.expander("💡 AI Cleaning Suggestions"):
                    if st.button("Analyze Data Health"):
                        with st.spinner("🤖 Analyzing..."):
                            string_io = StringIO()
                            df_processed.info(buf=string_io)
                            prompt = f"Analyze this dataframe info and suggest 3 critical cleaning steps:\n{string_io.getvalue()}"
                            try:
                                response = model.generate_content(prompt)
                                st.info(response.text)
                            except: st.error("AI not configured.")

                with st.expander("➕ Add New Column"):
                    new_col_name = st.text_input("New Column Name:")
                    new_col_val = st.text_input("Default Value (Applies to all rows):")
                    if st.button("Add Column"):
                        if new_col_name:
                            add_to_history(df_processed)
                            st.session_state.df_processed[new_col_name] = new_col_val
                            st.success(f"✅ Column '{new_col_name}' added.")
                            st.rerun()

                with st.expander("🗑️ Drop Unwanted Columns"):
                    cols_to_drop = st.multiselect("Select columns to remove:", df_processed.columns)
                    if st.button("Drop Selected Columns"):
                        add_to_history(df_processed)
                        st.session_state.df_processed = df_processed.drop(columns=cols_to_drop)
                        st.success("✅ Columns dropped.")
                        st.rerun()

                with st.expander("🔄 Change Data Types"):
                    col_to_change = st.selectbox("Column:", df_processed.columns, key="dtype_col")
                    new_type = st.selectbox("New Type:", ["string", "int", "float", "datetime"])
                    if st.button("Apply Type Change"):
                        add_to_history(df_processed)
                        try:
                            if new_type == "datetime": st.session_state.df_processed[col_to_change] = pd.to_datetime(df_processed[col_to_change])
                            else: st.session_state.df_processed[col_to_change] = df_processed[col_to_change].astype(new_type)
                            st.success("✅ Type Updated")
                            st.rerun()
                        except Exception as e: st.error(f"Error: {e}")

            with c2:
                st.caption("🧼 Content Cleaning")
                with st.expander("💧 Handle Missing Values"):
                    if missing_cells > 0:
                        st.dataframe(df_processed.isnull().sum()[df_processed.isnull().sum() > 0], height=100)
                        method = st.selectbox("Method:", ["Drop Rows", "Fill Mean (Numeric Only)", "Fill Mode (All)", "Fill Zero/Unknown"])
                        if st.button("Apply Fix"):
                            add_to_history(df_processed)
                            if method.startswith("Drop"): st.session_state.df_processed = df_processed.dropna()
                            elif method.startswith("Fill Mean"):    
                                num = df_processed.select_dtypes(include=np.number).columns
                                st.session_state.df_processed[num] = df_processed[num].fillna(df_processed[num].mean())
                            elif method.startswith("Fill Mode"):    
                                for c in df_processed.columns:
                                    if df_processed[c].isnull().any():
                                        st.session_state.df_processed[c] = df_processed[c].fillna(df_processed[c].mode()[0])
                            elif method.startswith("Fill Zero"):
                                st.session_state.df_processed = df_processed.fillna(0)
                            st.success("✅ Missing values handled")
                            st.rerun()
                    else:
                        st.success("✨ No missing values found!")

                with st.expander("🧹 Remove Duplicates"):
                    if duplicate_rows > 0:
                        st.warning(f"Found {duplicate_rows} duplicates.")
                        if st.button("Remove All Duplicates"):
                            add_to_history(df_processed)
                            st.session_state.df_processed = df_processed.drop_duplicates()
                            st.rerun()
                    else:
                        st.success("✨ No duplicates found.")

                with st.expander("✏️ Text & String Ops"):
                    text_col = st.selectbox("Text Column:", df_processed.select_dtypes(include=['object']).columns)
                    action = st.selectbox("Action:", ["Uppercase", "Lowercase", "Trim Spaces"])
                    if st.button("Apply Text Fix"):
                        add_to_history(df_processed)
                        if action == "Uppercase": st.session_state.df_processed[text_col] = df_processed[text_col].str.upper()
                        elif action == "Lowercase": st.session_state.df_processed[text_col] = df_processed[text_col].str.lower()
                        elif action == "Trim Spaces": st.session_state.df_processed[text_col] = df_processed[text_col].str.strip()
                        st.rerun()

                with st.expander("🔍 Find and Replace"):
                    fr_col = st.selectbox("Column:", df_processed.columns, key="fr_col")
                    find_txt = st.text_input("Find:")
                    replace_txt = st.text_input("Replace with:")
                    if st.button("Replace All"):
                        add_to_history(df_processed)
                        st.session_state.df_processed[fr_col] = df_processed[fr_col].replace(find_txt, replace_txt)
                        st.success("✅ Replaced")
                        st.rerun()

            st.markdown("---")
            st.subheader("📝 Live Data Editor")
            editor_key = f"editor_{len(st.session_state.history)}"
            edited_df = st.data_editor(df_processed, num_rows="dynamic", use_container_width=True, key=editor_key)
            if not edited_df.equals(df_processed):
                 add_to_history(df_processed)
                 st.session_state.df_processed = edited_df
                 st.rerun()

            st.markdown("---")
            st.subheader("💾 Download Results")
            d_col1, d_col2, d_col3 = st.columns([1, 1, 2])
            with d_col1:
                st.download_button("📥 Download CSV", data=convert_df_to_csv(df_processed), file_name="clean_data.csv", mime="text/csv", use_container_width=True)
            with d_col2:
                st.download_button("📥 Download Excel", data=convert_df_to_excel(df_processed), file_name="clean_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            
            if len(st.session_state.history) > 1:
                with d_col3:
                    if st.button("↩️ Undo Last Action", use_container_width=True):
                        st.session_state.history.pop()
                        st.session_state.df_processed = st.session_state.history[-1].copy()
                        st.rerun()

        # === TAB 3: ONE-CLICK DASHBOARD (PRESERVED & INTEGRATED) ===
        with tab3:
            render_ocd_dashboard(df_processed)

        # === TAB 4: PRO ML STUDIO V2.0 (UPGRADED) ===
        with tab4:
            render_ml_studio(df_processed)

    else:
        st.info("Please upload a CSV or Excel file to begin analysis.")
        if 'df_processed' in st.session_state:
            del st.session_state.df_processed
            del st.session_state.history

# --- SIDEBAR AND MAIN APP LOGIC ---
with st.sidebar:
    st.title("Decisyn AI")
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=180)
    else:
        st.markdown("### 🤖") 
        
    st.markdown("---")
    app_mode = st.radio("Navigation", ["AI Code Generator", "Excel Analyzer"], label_visibility="collapsed")
    st.markdown("---")
    if GEMINI_CONFIGURED:
        st.success("Status: **Online Mode**")
    else:
        st.error("Status: **Offline Mode**\n\nAPI Key not found.")

# --- UI FUNCTION FOR CODE GENERATOR ---
def page_code_generator():
    st.header("Translate Natural Language to Code")
    st.markdown("Enter a description, and AI will generate the Python code.")
    user_prompt = st.text_area("Your request:", height=150, placeholder="e.g., create a dataframe...")
    
    if st.button("Generate Code", type="primary"):
        if 'GEMINI_CONFIGURED' not in globals() or not GEMINI_CONFIGURED:
            st.error("Gemini AI is not configured. Please check your API Key.")
            return
            
        if user_prompt:
            with st.spinner("🤖 Calling Gemini AI..."):
                try:
                    full_prompt = f"Generate ONLY raw Python code for: {user_prompt}"
                    response = model.generate_content(full_prompt)
                    st.code(response.text, language='python')
                except Exception as e:
                    st.error(f"Error: {e}")

                    st.markdown("### ⚡ Instant Dataset Generator")

# User Input
user_prompt = st.text_area("Describe the data you need:", height=100)

# The Button
if st.button("Generate Excel File"):
    if not user_prompt:
        st.warning("Please type a description first.")
    else:
        with st.spinner("🤖 AI is generating your file..."):
            # Because we defined the function at the TOP, this now works!
            excel_file, status = generate_excel_from_prompt(user_prompt)
            
            if excel_file:
                st.success("✅ Done!")
                st.download_button(
                    label="📥 Download .xlsx",
                    data=excel_file,
                    file_name="generated_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.error(status)

if app_mode == "AI Code Generator":
    page_code_generator()
elif app_mode == "Excel Analyzer":
    page_data_analyzer()
