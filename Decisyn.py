import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
import os
import google.generativeai as genai
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

# --- ML Libraries (Expanded) ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Decisyn AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ==========================================
# 🎨 DYNAMIC THEME ENGINE
# ==========================================

def apply_theme_style(theme_name):
    # Modified default for better dashboard clarity and removed specific line issues.
    themes = {
        "Interactive Light": {
             "bg": "#f0f2f6", "card_bg": "#ffffff", "text": "#1f2937", "border": "#e5e7eb", "accent": "#4c7cff", "shadow": "0 5px 15px rgba(0,0,0,0.1)"
        },
        "Dark Pro": {
            "bg": "#0E1117", "card_bg": "#1E1E1E", "text": "#FAFAFA", "border": "#333", "accent": "#00CC96", "shadow": "0 4px 6px rgba(0,0,0,0.3)"
        },
        "Light Glass": {
            "bg": "#FFFFFF", "card_bg": "#F0F2F6", "text": "#31333F", "border": "#E6E6EA", "accent": "#FF4B4B", "shadow": "0 2px 4px rgba(0,0,0,0.1)"
        },
        "Neon Cyber": {
            "bg": "#000000", "card_bg": "#090909", "text": "#00FF00", "border": "#00FF00", "accent": "#FF00FF", "shadow": "0 0 15px rgba(0,255,0,0.4)"
        },
        "Sunset Synth": {
            "bg": "#2b102f", "card_bg": "#411b4b", "text": "#ffd1dc", "border": "#ff9900", "accent": "#ff0055", "shadow": "0 4px 15px rgba(255, 153, 0, 0.3)"
        },
        "Midnight Blue": {
            "bg": "#021024", "card_bg": "#052659", "text": "#7DA0CA", "border": "#5483B3", "accent": "#C1E8FF", "shadow": "0 4px 10px rgba(0,0,0,0.5)"
        },
        "Forest Glass": {
            "bg": "#1a2f23", "card_bg": "rgba(255,255,255,0.1)", "text": "#e0f2e9", "border": "#4ade80", "accent": "#22c55e", "shadow": "0 4px 12px rgba(0,0,0,0.2)"
        }
    }
    t = themes.get(theme_name, themes["Interactive Light"])
    accent_color = t['accent']
    
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
        .metric-card {{
            background-color: {t['card_bg']}; border-radius: 15px; padding: 15px 20px;
            box-shadow: {t['shadow']}; border: 1px solid {t['border']}; margin-bottom: 15px;
            transition: all 0.3s ease-in-out; 
            border-left: 5px solid {accent_color}; /* Colorful accent border */
            position: relative; /* For tooltip */
        }}
        .metric-card:hover {{ transform: translateY(-5px); border-color: {accent_color}; box-shadow: 0 8px 20px rgba(0,0,0,0.2); }}
        .metric-label {{ font-size: 13px; color: {t['text']}; opacity: 0.8; margin-bottom: 5px; text-transform: uppercase; font-weight: 600; }}
        .metric-value {{ font-size: 28px; font-weight: 700; color: {accent_color}; }}
        .ocd-chart-card {{
            background-color: {t['card_bg']}; border-radius: 10px; padding: 10px;
            box-shadow: {t['shadow']}; margin-bottom: 15px;
            overflow: hidden; /* Fixes potential plot overlap */
        }}

        /* --- STYLING TABS AS BUTTON BOXES --- */
        [data-baseweb="tab-list"] {{
            gap: 15px; /* Spacing between "buttons" */
        }}
        button[data-baseweb="tab"] {{
            background-color: {t['card_bg']} !important;
            border: 2px solid {t['border']} !important;
            border-radius: 10px !important;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            padding: 10px 20px;
            text-shadow: none;
            color: {t['text']} !important;
            border-bottom: 0px !important; /* Remove default tab underline */
        }}
        button[data-baseweb="tab"]:hover {{
            box-shadow: 4px 4px 10px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        /* Highlight active tab */
        button[data-baseweb="tab"][aria-selected="true"] {{
            border-color: {accent_color} !important;
            color: {accent_color} !important; 
            background-color: #e8f0ff !important; /* Light accent background for active tab */
            border-left: 5px solid {accent_color} !important;
        }}
        /* Sidebar Cleanup */
        [data-testid="stSidebar"] {{ border-right: 1px solid {t['border']}; }}
        </style>
    """, unsafe_allow_html=True)
    
    if theme_name in ["Light Glass", "Interactive Light"]: return "plotly_white"
    else: return "plotly_dark"

# --- Custom KPI Renderer for Hover ---
def render_kpi_with_hover(df_filtered, col, accent_color):
    if df_filtered.empty or col not in df_filtered.columns or df_filtered[col].isnull().all():
        val = 0
        max_val = 0
        min_val = 0
    else:
        val = df_filtered[col].sum()
        max_val = df_filtered[col].max()
        min_val = df_filtered[col].min()

    st.markdown(f"""
    <div class="metric-card" title="Highest: {max_val:,.0f} | Lowest: {min_val:,.0f}">
        <div class="metric-label">{col}</div>
        <div class="metric-value">{val:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 🤖 PRO ML STUDIO (RESTRUCTURED AND FIXED)
# ==========================================
def render_ml_studio(df):
    st.markdown("## 🧠 Pro ML Studio (Advanced Analytics)")
    
    # Define tabs
    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["🔮 Seasonal Forecast", "🎯 Supervised (Train & Predict)", "🧬 Unsupervised (3D & Hierarchy)"])

    # --- TAB 1: FORECAST ---
    with ml_tab1:
        st.subheader("📈 Time-Series Forecasting")
        date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if date_cols and num_cols:
            c1, c2 = st.columns(2)
            date_col = c1.selectbox("Date Column", date_cols)
            target_col = c2.selectbox("Target Metric", num_cols)
            if st.button("Generate Forecast", key="forecast_btn"):
                df_g = df.groupby(date_col)[target_col].sum().reset_index().sort_values(date_col)
                df_g['idx'] = np.arange(len(df_g))
                model = LinearRegression()
                model.fit(df_g[['idx']], df_g[target_col])
                
                future_idx = np.arange(len(df_g), len(df_g)+12).reshape(-1, 1)
                preds = model.predict(future_idx)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_g[date_col], y=df_g[target_col], name='Historical', mode='lines', line=dict(color='#4c7cff')))
                fig.add_trace(go.Scatter(x=pd.date_range(df_g[date_col].max(), periods=13, freq='M')[1:], y=preds, name='Forecast', mode='lines', line=dict(color='#00b894', dash='dash')))
                fig.update_layout(title="Linear Trend Forecast", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Needs both DateTime and Numerical columns.")

    # --- TAB 2: SUPERVISED (FIXED WITH MANUAL PREDICTION) ---
    with ml_tab2:
        st.subheader("🎯 Predictive Modeling & Testing")
        
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Target Variable", df.columns, key='sup_target')
        feats = c2.multiselect("Features", [c for c in df.columns if c != target])
        algo = c3.selectbox("Algorithm", ["Linear/Logistic Regression", "Random Forest", "Decision Tree"])
        
        # --- FIX: Problem Type Selector ---
        # Initial guess based on data
        is_numeric_target = pd.api.types.is_numeric_dtype(df[target])
        initial_guess = "Regression" if is_numeric_target and df[target].nunique() > 5 else "Classification"
        
        problem_type = st.radio(
            "Force Problem Type:", 
            ["Regression", "Classification"], 
            index=0 if initial_guess == "Regression" else 1, 
            horizontal=True, 
            key='prob_type_selector'
        )
        
        if st.button("🚀 Train Model", key="train_btn"):
            if feats:
                df_model = df[[target] + feats].dropna()
                X = pd.get_dummies(df_model[feats], drop_first=True)
                y = df_model[target]
                
                # Check if the selection makes sense
                if problem_type == "Regression" and not is_numeric_target:
                    st.error("Cannot run Regression on non-numeric target variable.")
                    return
                
                is_reg = (problem_type == "Regression")
                
                # Data Preparation (Scaling and Splitting)
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X) # Scale before splitting for features
                
                # Store encoder if classification
                if not is_reg:
                    # FIX: Encode the FULL target variable before splitting
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y)
                    y_train, y_test = train_test_split(y_encoded, test_size=0.2, random_state=42)
                else:
                    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

                X_train, X_test = train_test_split(X_sc, test_size=0.2, random_state=42)

                # Model Selection
                if is_reg:
                    if algo == "Random Forest": model = RandomForestRegressor()
                    elif algo == "Decision Tree": model = DecisionTreeRegressor()
                    else: model = LinearRegression()
                else:
                    if algo == "Random Forest": model = RandomForestClassifier()
                    elif algo == "Decision Tree": model = DecisionTreeClassifier()
                    else: model = LogisticRegression(max_iter=1000)
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Save model components for manual prediction
                st.session_state['trained_model'] = model
                st.session_state['model_features'] = X.columns.tolist() # Store OHE columns
                st.session_state['model_type'] = problem_type
                st.session_state['model_scaler'] = scaler
                st.session_state['original_feats'] = feats
                if not is_reg: st.session_state['le'] = le
                
                score = r2_score(y_test, preds) if is_reg else accuracy_score(y_test, preds)
                st.success(f"Model Trained Successfully! {problem_type} Score: **{score:.2%}**")

        # --- Manual Prediction Input ---
        st.markdown("---")
        if 'trained_model' in st.session_state:
            st.markdown("#### 🔮 Manual Prediction Input")
            
            model = st.session_state['trained_model']
            original_feats = st.session_state['original_feats']
            input_data = {}
            
            # Dynamic Input Fields
            cols = st.columns(3)
            for idx, col_name in enumerate(original_feats):
                with cols[idx % 3]:
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        input_data[col_name] = st.number_input(f"{col_name}", value=float(df[col_name].mean()), key=f'pred_in_{col_name}')
                    else:
                        # For simplicity, skip complex OHE features in manual input, focus on simple numeric/binary inputs
                        st.warning(f"Feature '{col_name}' requires encoding.")

            if st.button("✨ Predict Outcome", key='pred_outcome_btn'):
                try:
                    # Collect input and create a DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Align and Scale input data
                    # Note: We must ensure the input columns match the OHE columns used during training.
                    # Since we only allowed numeric inputs for simplicity, we directly scale and predict.
                    input_sc = st.session_state['model_scaler'].transform(input_df) 
                    
                    val = model.predict(input_sc)[0]
                    
                    st.balloons()
                    if st.session_state['model_type'] == 'Classification':
                        le = st.session_state.get('le')
                        val_decoded = le.inverse_transform([int(val)])[0] if le else f"Class Index {int(val)}"
                        st.info(f"Predicted Class: **{val_decoded}**")
                    else:
                        st.info(f"Predicted Value: **{val:,.2f}**")
                except Exception as e:
                    st.error(f"Prediction Error: Inputs must match trained feature structure. Ensure all relevant numeric fields are filled.")


    # --- TAB 3: UNSUPERVISED (3D & HIERARCHY) ---
    with ml_tab3:
        st.subheader("🧬 Clustering (K-Means & Hierarchical)")
        
        # K-Means 3D Plot
        st.markdown("#### 3D K-Means Visualization")
        num_features = df.select_dtypes(include=['number']).columns.tolist()
        feats_3d = st.multiselect("Select 3 Features for K-Means", num_features, default=num_features[:3] if len(num_features)>=3 else [], key='km_feats')
        k = st.slider("Number of Clusters (k)", 2, 8, 3, key='km_k')
        
        if st.button("Run 3D K-Means", key='kmeans_3d_btn') and len(feats_3d) == 3:
            X = df[feats_3d].dropna()
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(StandardScaler().fit_transform(X))
            X['Cluster'] = clusters.astype(str)
            
            fig = px.scatter_3d(X, x=feats_3d[0], y=feats_3d[1], z=feats_3d[2], 
                              color='Cluster', title=f"3D K-Means (k={k})", 
                              color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
        elif len(feats_3d) != 3 and st.button("Check 3D Feats", key='check_3d'):
            st.warning("Please select exactly 3 features for the 3D chart.")

        st.markdown("---")

        # Hierarchical Clustering (Dendrogram)
        st.markdown("#### Interactive Dendrogram")
        st.caption("Visualizes data hierarchy. Sampling first 500 rows for performance.")
        
        feats_hier = st.multiselect("Select Features for Hierarchy", num_features, default=num_features[:5] if len(num_features)>=5 else [], key='hier_feats')
        threshold = st.slider("Cluster Threshold (Cutoff)", 0.0, 100.0, 50.0, key='hier_thresh')
        
        if st.button("Generate Dendrogram", key='dendro_btn') and feats_hier:
            X_hier = df[feats_hier].dropna().head(500)
            X_scaled = StandardScaler().fit_transform(X_hier)
            
            linked = shc.linkage(X_scaled, method='ward')
            
            fig_plt, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Hierarchical Clustering Dendrogram")
            shc.dendrogram(linked, orientation='top', p=threshold, truncate_mode='lastp',
                           color_threshold=threshold, ax=ax)
            ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.1f}')
            ax.legend()
            st.pyplot(fig_plt)
            st.markdown("Showing cluster hierarchy. The red line cuts off clusters at the selected **Threshold**.")
            

# ==========================================
# 🚀 UPDATED: INTERACTIVE DASHBOARD (OCD)
# ==========================================
def render_ocd_dashboard(df):
    # --- Sidebar Configuration (Enhanced Slicers) ---
    st.sidebar.header("🎨 Dashboard Settings")
    
    # 1. Theme (Sidebar)
    theme_options = ["Interactive Light", "Dark Pro", "Light Glass", "Neon Cyber", "Sunset Synth", "Midnight Blue", "Forest Glass"]
    selected_theme = st.sidebar.selectbox("Theme", theme_options, index=0)
    plotly_template = apply_theme_style(selected_theme)
    
    # Define colors based on selected theme 
    colors = px.colors.qualitative.Bold
    if selected_theme == "Interactive Light":
        accent_color = '#4c7cff'
    elif selected_theme == "Neon Cyber":
        accent_color = '#00FF00'
    else:
        accent_color = colors[0]

    # 2. KPI Selector (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 KPI Metrics")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    default_kpis = num_cols[:4] if len(num_cols) >= 4 else num_cols
    selected_kpis = st.sidebar.multiselect("Select KPIs to Display", num_cols, default=default_kpis)
    
    # 3. Filters/Slicers (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌪️ Global Slicers")
    df_filtered = df.copy()
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
    
    # Categorical Slicers
    for i, col in enumerate(cat_cols[:3]): 
        f_val = st.sidebar.multiselect(f"Filter by **{col}**", df[col].unique(), key=f'cat_slicer_{i}')
        if f_val:
            df_filtered = df_filtered[df_filtered[col].isin(f_val)]
            
    # Date Slicer
    if date_cols:
        date_col = date_cols[0]
        min_d, max_d = df[date_col].min(), df[date_col].max()
        
        with st.sidebar.expander(f"📅 Date Range ({date_col})"):
            dates = st.date_input("Select Range", [min_d, max_d], min_value=min_d, max_value=max_d)
            if len(dates) == 2:
                df_filtered = df_filtered[(df_filtered[date_col].dt.date >= dates[0]) & (df_filtered[date_col].dt.date <= dates[1])]
    
    # 4. Download (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download Filtered Data",
        data=convert_df_to_excel(df_filtered),
        file_name="dashboard_data_ocd.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # --- MAIN DASHBOARD LAYOUT ---
    st.markdown("## 🚀 One-Click Business Dashboard")

    # 1. KPI SECTION (4xN LAYOUT & HOVER)
    if selected_kpis:
        st.markdown("#### Key Performance Indicators")
        
        for i in range(0, len(selected_kpis), 4):
            kpi_cols = st.columns(4)
            for j, col in enumerate(selected_kpis[i:i+4]):
                with kpi_cols[j]:
                    render_kpi_with_hover(df_filtered, col, accent_color)
    
    # 2. 3x3 CHART GRID
    st.markdown("### 📈 Visual Analytics Grid")
    
    chart_configs = [
        ("Time Trend", date_cols, num_cols),
        ("Category Breakdown", cat_cols, num_cols),
        ("Top N Bar Chart", cat_cols, num_cols),
        ("Correlation Scatter", num_cols, num_cols),
        ("Distribution Histogram", num_cols, None),
        ("Box Plot Outliers", num_cols, cat_cols),
        ("Sunburst Hierarchy", cat_cols, num_cols),
        ("Bubble Chart", num_cols, num_cols),
        ("Map/Table Fallback", cat_cols, num_cols), 
    ]
    
    chart_idx = 0
    for r in range(3):
        cols = st.columns(3)
        for c in range(3):
            title, x_data, y_data = chart_configs[chart_idx]
            
            with cols[c]:
                st.markdown('<div class="ocd-chart-card">', unsafe_allow_html=True)
                st.markdown(f"**{chart_idx + 1}. {title}**")
                
                # Plotting Logic
                if x_data and num_cols:
                    try:
                        # Dynamic Expander for Chart Edit Options
                        with st.expander("⚙️ Edit Chart"):
                             c_x = st.selectbox("X-Axis", x_data, index=0, key=f'c{chart_idx}_x')
                             c_y = st.selectbox("Y-Axis", num_cols, index=0, key=f'c{chart_idx}_y')
                             c_type = st.selectbox("Type", ["Area", "Pie", "Bar", "Scatter"], key=f'c{chart_idx}_t')

                        if title == "Time Trend" and date_cols:
                            df_plot = df_filtered.groupby(date_cols[0])[num_cols[0]].sum().reset_index()
                            fig = px.area(df_plot, x=date_cols[0], y=num_cols[0], color_discrete_sequence=colors, template=plotly_template)
                        
                        elif title == "Category Breakdown" and cat_cols:
                            df_plot = df_filtered.groupby(cat_cols[0])[num_cols[0]].sum().reset_index()
                            fig = px.pie(df_plot, values=num_cols[0], names=cat_cols[0], hole=0.5, color_discrete_sequence=colors, template=plotly_template)
                        
                        elif title == "Correlation Scatter" and len(num_cols) >= 2:
                            fig = px.scatter(df_filtered, x=num_cols[0], y=num_cols[1], color=cat_cols[0] if cat_cols else None, color_discrete_sequence=colors, template=plotly_template)
                        
                        elif title == "Distribution Histogram":
                            fig = px.histogram(df_filtered, x=num_cols[0], color_discrete_sequence=[accent_color], template=plotly_template)
                        
                        elif title == "Box Plot Outliers" and cat_cols:
                            fig = px.box(df_filtered, x=cat_cols[0], y=num_cols[0], color=cat_cols[0], color_discrete_sequence=colors, template=plotly_template)

                        elif title == "Sunburst Hierarchy" and len(cat_cols) >= 2:
                             fig = px.sunburst(df_filtered, path=cat_cols[:2], values=num_cols[0], color_discrete_sequence=colors, template=plotly_template)
                        
                        elif title == "Bubble Chart" and len(num_cols) >= 3:
                             fig = px.scatter(df_filtered, x=num_cols[0], y=num_cols[1], size=num_cols[2], color=cat_cols[0] if cat_cols else None, color_discrete_sequence=colors, template=plotly_template)

                        else: 
                            fig = px.bar(df_filtered, x=cat_cols[0] if cat_cols else None, y=num_cols[0], color_discrete_sequence=colors, template=plotly_template)

                        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=250, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        
                    except Exception as e:
                        st.info(f"Configuration Missing or Data Error.")
                        
                else:
                    st.info("Insufficient data for analysis.")
                st.markdown('</div>', unsafe_allow_html=True) 
            
            chart_idx += 1


# --- UI FUNCTION FOR CODE GENERATOR ---
def page_code_generator():
    st.header("Translate Natural Language to Code")
    st.markdown("Enter a description, and AI will generate the Python code.")
    user_prompt = st.text_area("Your request:", height=150, placeholder="e.g., create a dataframe...")
    if st.button("Generate Code", type="primary"):
        if not GEMINI_CONFIGURED:
            st.error("Gemini AI is not configured.")
            return
        if user_prompt:
            with st.spinner("🤖 Calling Gemini AI..."):
                try:
                    full_prompt = f"Generate ONLY raw Python code for: {user_prompt}"
                    response = model.generate_content(full_prompt)
                    st.code(response.text, language='python')
                except Exception as e:
                    st.error(f"Error: {e}")

# --- UI FUNCTION FOR DATA ANALYZER ---
def page_data_analyzer():
    st.header("Analyze, Clean & Visualize Your Data")
    st.markdown("Upload a CSV or Excel file to get instant insights, clean data, and build interactive charts.")
    
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
            keys_to_clear = ['ml_model', 'ml_score', 'cluster_df', 'ai_summary']
            for k in keys_to_clear:
                if k in st.session_state: del st.session_state[k]
        
        df_processed = st.session_state.df_processed
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizer", "🛠️ Data Explorer & Cleaner", "🚀 One-Click Dashboard", "🤖 Pro ML Studio"])
        
        # === TAB 1: VISUALIZER (Edit Chart Options Restored) ===
        with tab1:
            st.subheader("Interactive Visualization")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("##### Chart Options")
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
                        fig.update_layout(title_text=chart_title, template="plotly_dark")
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
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate chart: {e}")

        # === TAB 2: EXPLORER & CLEANER (ORIGINAL) ===
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
                                code_response = model.generate_content(prompt)
                                generated_code = code_response.text.strip().replace("python", "").replace("```", "")
                                
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

        # === TAB 3: ONE-CLICK DASHBOARD (UPDATED) ===
        with tab3:
            render_ocd_dashboard(df_processed)

        # === TAB 4: PRO ML STUDIO ===
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

if app_mode == "AI Code Generator":
    page_code_generator()
elif app_mode == "Excel Analyzer":
    page_data_analyzer()
