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

# --- ML Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Decisyn AI",
    page_icon="🤖",
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
    themes = {
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
    t = themes.get(theme_name, themes["Dark Pro"])
    
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
        .metric-card {{
            background-color: {t['card_bg']}; border-radius: 15px; padding: 15px 20px;
            box-shadow: {t['shadow']}; border: 1px solid {t['border']}; margin-bottom: 15px;
            transition: all 0.3s ease-in-out; backdrop-filter: blur(10px);
        }}
        .metric-card:hover {{ transform: translateY(-5px); border-color: {t['accent']}; box-shadow: 0 8px 20px rgba(0,0,0,0.6); }}
        .metric-label {{ font-size: 13px; color: {t['text']}; opacity: 0.8; margin-bottom: 5px; text-transform: uppercase; font-weight: 600; }}
        .metric-value {{ font-size: 28px; font-weight: 700; color: {t['text']}; }}
        /* Sidebar Cleanup */
        [data-testid="stSidebar"] {{ border-right: 1px solid {t['border']}; }}
        </style>
    """, unsafe_allow_html=True)
    
    if theme_name == "Light Glass": return "plotly_white"
    elif theme_name in ["Neon Cyber", "Sunset Synth", "Midnight Blue"]: return "plotly_dark"
    else: return "plotly_dark"

# ==========================================
# 🤖 PRO ML STUDIO
# ==========================================
def render_ml_studio(df):
    st.markdown("## 🤖 Pro ML Studio")
    st.markdown("Advanced analytics environment for MBA & Data Science.")
    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["🔮 Seasonal Forecast", "🎯 Supervised (Prediction)", "🧬 Unsupervised (Clustering)"])

    with ml_tab1:
        st.subheader("Time-Series Forecasting")
        date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if date_cols:
            c1, c2 = st.columns(2)
            with c1: date_col = st.selectbox("Date Column", date_cols)
            with c2: target_col = st.selectbox("Target", num_cols)
            if st.button("Generate Forecast"):
                df_g = df.groupby(date_col)[target_col].sum().reset_index().sort_values(date_col)
                df_g['idx'] = np.arange(len(df_g))
                model = LinearRegression()
                model.fit(df_g[['idx']], df_g[target_col])
                future_idx = np.arange(len(df_g), len(df_g)+12).reshape(-1, 1)
                preds = model.predict(future_idx)
                fig = px.line(y=np.concatenate([df_g[target_col], preds]), title="Trend Forecast")
                st.plotly_chart(fig, use_container_width=True)

    with ml_tab2:
        st.subheader("Prediction")
        target = st.selectbox("Target Variable", df.columns)
        feats = st.multiselect("Features", [c for c in df.columns if c != target])
        if st.button("Train Model") and feats:
            X = pd.get_dummies(df[feats].dropna())
            y = df.loc[X.index, target]
            if pd.api.types.is_numeric_dtype(y):
                m = LinearRegression()
                m.fit(X, y)
                st.metric("R2 Score", f"{r2_score(y, m.predict(X)):.2f}")
            else:
                le = LabelEncoder()
                y = le.fit_transform(y)
                m = LogisticRegression()
                m.fit(X, y)
                st.metric("Accuracy", f"{accuracy_score(y, m.predict(X)):.2%}")

    with ml_tab3:
        st.subheader("Clustering")
        feats = st.multiselect("Cluster Features", df.select_dtypes(include=['number']).columns)
        k = st.slider("Clusters (k)", 2, 8, 3)
        if st.button("Run K-Means") and feats:
            X = df[feats].dropna()
            kmeans = KMeans(n_clusters=k)
            df['Cluster'] = kmeans.fit_predict(X)
            st.plotly_chart(px.scatter(df, x=feats[0], y=feats[1], color='Cluster'))

# ==========================================
# 🚀 UPDATED: INTERACTIVE DASHBOARD
# ==========================================
def render_ocd_dashboard(df):
    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("🎨 Dashboard Settings")
    
    # 1. Theme (Sidebar)
    selected_theme = st.sidebar.selectbox("Theme", ["Dark Pro", "Light Glass", "Neon Cyber", "Sunset Synth", "Midnight Blue", "Forest Glass"])
    plotly_template = apply_theme_style(selected_theme)
    
    # Colors
    if selected_theme == "Neon Cyber":
        colors = ['#FF00FF', '#00FF00', '#00FFFF', '#FFFF00']
        bg_color = "#000000"
    elif selected_theme == "Sunset Synth":
        colors = px.colors.sequential.Plasma
        bg_color = "#2b102f"
    elif selected_theme == "Forest Glass":
        colors = px.colors.sequential.Greens
        bg_color = "#1a2f23"
    elif selected_theme == "Light Glass":
        colors = px.colors.qualitative.Bold
        bg_color = "#FFFFFF"
    else:
        colors = px.colors.qualitative.Vivid
        bg_color = "rgba(0,0,0,0)"

    # 2. KPI Selector (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 KPI Management")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_kpis = []
    if num_cols:
        selected_kpis = st.sidebar.multiselect("Select KPIs to Display", num_cols, default=num_cols[:4] if len(num_cols) >= 4 else num_cols)

    # 3. Filters (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌪️ Global Filters")
    df_filtered = df.copy()
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        f1 = st.sidebar.multiselect(f"By {cat_cols[0]}", df[cat_cols[0]].unique())
        if f1: df_filtered = df_filtered[df_filtered[cat_cols[0]].isin(f1)]
        if len(cat_cols) > 1:
            f2 = st.sidebar.multiselect(f"By {cat_cols[1]}", df[cat_cols[1]].unique())
            if f2: df_filtered = df_filtered[df_filtered[cat_cols[1]].isin(f2)]

    date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
    if date_cols:
        min_d, max_d = df[date_cols[0]].min(), df[date_cols[0]].max()
        dates = st.sidebar.date_input("Date Range", [min_d, max_d], min_value=min_d, max_value=max_d)
        if len(dates) == 2:
            df_filtered = df_filtered[(df_filtered[date_cols[0]].dt.date >= dates[0]) & (df_filtered[date_cols[0]].dt.date <= dates[1])]

    # 4. Download (Sidebar)
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download Filtered Data",
        data=convert_df_to_excel(df_filtered),
        file_name="dashboard_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # --- MAIN DASHBOARD LAYOUT ---
    st.markdown("## 🚀 Interactive Business Dashboard")
    
    # KPI Section
    if selected_kpis:
        kpi_cols = st.columns(len(selected_kpis))
        for i, col in enumerate(selected_kpis):
            val = df_filtered[col].sum()
            with kpi_cols[i]:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">{col}</div><div class="metric-value">{val:,.0f}</div></div>""", unsafe_allow_html=True)
    
    st.divider()

    # Charts Grid (2x2) - Controls Hidden in Expanders
    st.subheader("📈 Visual Analytics")
    r1_c1, r1_c2 = st.columns(2)
    r2_c1, r2_c2 = st.columns(2)
    
    # Chart 1: Category Breakdown
    with r1_c1:
        st.markdown("**1. Category Breakdown**")
        if cat_cols and num_cols:
            with st.expander("⚙️ Edit Chart"):
                c1_type = st.selectbox("Type", ["Donut", "Pie", "Sunburst", "Treemap"], key="c1_t")
                c1_group = st.selectbox("Group By", cat_cols, key="c1_g")
                c1_val = st.selectbox("Value", num_cols, key="c1_v")
            
            data = df_filtered.groupby(c1_group)[c1_val].sum().reset_index()
            if c1_type == "Donut": fig1 = px.pie(data, values=c1_val, names=c1_group, hole=0.5, color_discrete_sequence=colors, template=plotly_template)
            elif c1_type == "Pie": fig1 = px.pie(data, values=c1_val, names=c1_group, color_discrete_sequence=colors, template=plotly_template)
            elif c1_type == "Sunburst": fig1 = px.sunburst(df_filtered, path=cat_cols[:2] if len(cat_cols) > 1 else [c1_group], values=c1_val, color=c1_val, template=plotly_template)
            else: fig1 = px.treemap(df_filtered, path=[c1_group], values=c1_val, color=c1_val, template=plotly_template)
            
            fig1.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, height=350, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Time Trend
    with r1_c2:
        st.markdown("**2. Performance Over Time**")
        if date_cols and num_cols:
            with st.expander("⚙️ Edit Chart"):
                c2_type = st.selectbox("Type", ["Area Trend", "Line Trend", "Bar Trend"], key="c2_t")
                c2_val = st.selectbox("Metric", num_cols, index=0, key="c2_v")
            
            trend_data = df_filtered.groupby(date_cols[0])[c2_val].sum().reset_index()
            if c2_type == "Area Trend": fig2 = px.area(trend_data, x=date_cols[0], y=c2_val, color_discrete_sequence=colors, template=plotly_template)
            elif c2_type == "Line Trend": fig2 = px.line(trend_data, x=date_cols[0], y=c2_val, markers=True, color_discrete_sequence=colors, template=plotly_template)
            else: fig2 = px.bar(trend_data, x=date_cols[0], y=c2_val, color_discrete_sequence=colors, template=plotly_template)
            
            fig2.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, height=350, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig2, use_container_width=True)
        else: st.info("No Date Column Found")

    # Chart 3: Correlation
    with r2_c1:
        st.markdown("**3. Correlation Analysis**")
        if len(num_cols) >= 3:
            with st.expander("⚙️ Edit Chart"):
                c3_type = st.selectbox("Type", ["3D Scatter", "2D Scatter", "Bubble"], key="c3_t")
                x_val = st.selectbox("X Axis", num_cols, index=0, key="c3_x")
                y_val = st.selectbox("Y Axis", num_cols, index=1, key="c3_y")
                z_val = st.selectbox("Z / Size", num_cols, index=2, key="c3_z")
            
            if c3_type == "3D Scatter": fig3 = px.scatter_3d(df_filtered, x=x_val, y=y_val, z=z_val, color=cat_cols[0] if cat_cols else None, color_discrete_sequence=colors, template=plotly_template)
            elif c3_type == "2D Scatter": fig3 = px.scatter(df_filtered, x=x_val, y=y_val, color=cat_cols[0] if cat_cols else None, color_discrete_sequence=colors, template=plotly_template)
            else: fig3 = px.scatter(df_filtered, x=x_val, y=y_val, size=z_val, color=cat_cols[0] if cat_cols else None, color_discrete_sequence=colors, template=plotly_template)
            
            fig3.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, height=350, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig3, use_container_width=True)
        else: st.info("Need at least 3 numeric columns")

    # Chart 4: Distribution
    with r2_c2:
        st.markdown("**4. Distribution Analysis**")
        if num_cols:
            with st.expander("⚙️ Edit Chart"):
                c4_type = st.selectbox("Type", ["Funnel", "Histogram", "Box Plot"], key="c4_t")
                c4_val = st.selectbox("Metric", num_cols, index=0, key="c4_v")
                c4_group = st.selectbox("Group", cat_cols, index=0, key="c4_g") if cat_cols else None
            
            if c4_type == "Funnel" and c4_group:
                funnel_data = df_filtered.groupby(c4_group)[c4_val].sum().reset_index().sort_values(c4_val, ascending=False)
                fig4 = px.funnel(funnel_data, x=c4_val, y=c4_group, color_discrete_sequence=colors, template=plotly_template)
            elif c4_type == "Histogram": fig4 = px.histogram(df_filtered, x=c4_val, color=c4_group, color_discrete_sequence=colors, template=plotly_template)
            else: fig4 = px.box(df_filtered, x=c4_group, y=c4_val, color=c4_group, color_discrete_sequence=colors, template=plotly_template)
            
            fig4.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, height=350, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig4, use_container_width=True)

    # AI Summary
    st.divider()
    st.subheader("🤖 AI Executive Summary")
    if 'ai_summary' in st.session_state and st.session_state.get('ai_summary_file') == st.session_state.get('file_name'):
        st.markdown(st.session_state['ai_summary'])
        if st.button("🔄 Regenerate"):
             del st.session_state['ai_summary']
             st.rerun()
    else:
        if st.button("✨ Generate Insights"):
            if not GEMINI_CONFIGURED:
                st.error("Gemini AI is not configured.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        buffer = StringIO()
                        df.info(buf=buffer)
                        prompt = f"Analyze this data stats: {df.describe().to_string()}\nInfo: {buffer.getvalue()}\nProvide 3 key business insights."
                        response = model.generate_content(prompt)
                        st.session_state['ai_summary'] = response.text
                        st.session_state['ai_summary_file'] = st.session_state.get('file_name')
                        st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

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
        
        # === TAB 1: VISUALIZER (ORIGINAL) ===
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

        # === TAB 3: ONE-CLICK DASHBOARD ===
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
        st.image("logo.jpg", width=120)
    else:
        st.markdown("### 🤖") 
        
    st.markdown("<p style='font-size:x-small;'>Designed and developed by Dhananjay loks</p>", unsafe_allow_html=True)
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
