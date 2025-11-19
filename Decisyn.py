import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
import os
import google.generativeai as genai

# --- ML Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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
    st.error(f"Error configuring Gemini AI: {e}")
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
# 🤖 ML STUDIO FUNCTION
# ==========================================
def render_ml_studio(df):
    st.markdown("## 🤖 ML Studio")
    st.markdown("Build and Train Machine Learning models directly on your data.")

    task_type = st.radio("Select Task:", ["Supervised Learning (Prediction)", "Unsupervised Learning (Clustering)"], horizontal=True)

    if task_type == "Supervised Learning (Prediction)":
        st.subheader("Predictive Modeling")
        model_type = st.selectbox("Choose Algorithm:", ["Linear Regression (Predict Numbers)", "Logistic Regression (Predict Categories)"])
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select Target Variable (y):", df.columns)
        with col2:
            feature_cols = st.multiselect("Select Feature Variables (X):", [c for c in df.columns if c != target_col])

        if st.button("Train Model") and feature_cols and target_col:
            X = df[feature_cols].dropna()
            y = df.loc[X.index, target_col]
            
            X = pd.get_dummies(X, drop_first=True)
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            with st.spinner("Training Model..."):
                try:
                    if model_type.startswith("Linear"):
                        model_lr = LinearRegression()
                        model_lr.fit(X_train, y_train)
                        y_pred = model_lr.predict(X_test)
                        
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        
                        st.success("✅ Model Trained Successfully!")
                        m1, m2 = st.columns(2)
                        m1.metric("R² Score (Accuracy)", f"{r2:.2%}")
                        m2.metric("Mean Squared Error", f"{mse:.2f}")
                        
                        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title="Actual vs Predicted")
                        fig.add_shape(type="line", line=dict(dash="dash"), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
                        st.plotly_chart(fig, use_container_width=True)

                    elif model_type.startswith("Logistic"):
                        model_log = LogisticRegression(max_iter=1000)
                        model_log.fit(X_train, y_train)
                        y_pred = model_log.predict(X_test)
                        
                        acc = accuracy_score(y_test, y_pred)
                        st.success("✅ Model Trained Successfully!")
                        st.metric("Accuracy", f"{acc:.2%}")
                        
                        cm = confusion_matrix(y_test, y_pred)
                        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", labels=dict(x="Predicted", y="Actual", color="Count"))
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error Training Model: {e}")

    elif task_type == "Unsupervised Learning (Clustering)":
        st.subheader("K-Means Clustering")
        
        cluster_cols = st.multiselect("Select Features for Clustering:", df.select_dtypes(include=['number']).columns)
        k_value = st.slider("Number of Clusters (k):", 2, 10, 3)
        
        if st.button("Run Clustering") and cluster_cols:
            X = df[cluster_cols].dropna()
            
            with st.spinner("Clustering Data..."):
                try:
                    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X)
                    X['Cluster'] = clusters.astype(str)
                    
                    st.success(f"✅ Data grouped into {k_value} clusters!")
                    
                    if len(cluster_cols) >= 2:
                        fig = px.scatter(X, x=cluster_cols[0], y=cluster_cols[1], color='Cluster', title="Cluster Visualization", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Select at least 2 numerical columns to visualize clusters.")
                        
                    st.dataframe(X.head())
                except Exception as e:
                    st.error(f"Error in Clustering: {e}")

# ==========================================
# 🚀 ONE-CLICK DASHBOARD FUNCTION
# ==========================================
def render_ocd_dashboard(df):
    st.markdown("## 🚀 One-Click Dashboard")
    st.markdown("Automatic insights based on your dataset structure.")

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()

    st.subheader("📊 Key Performance Indicators")
    default_kpis = num_cols[:3] if len(num_cols) >= 3 else num_cols
    selected_kpis = st.multiselect("Select Metrics to Track:", num_cols, default=default_kpis)

    if selected_kpis:
        cols = st.columns(len(selected_kpis) + 1)
        cols[0].metric("Total Rows", f"{len(df):,}")
        for i, col_name in enumerate(selected_kpis):
            total_val = df[col_name].sum()
            cols[i+1].metric(f"Total {col_name}", f"{total_val:,.2f}")
    else:
        st.metric("Total Rows", len(df))

    st.divider()

    st.subheader("📈 Auto-Generated Visualizations")
    best_cat_col = next((col for col in cat_cols if 2 <= df[col].nunique() <= 20), None)
    col1, col2 = st.columns(2)

    if best_cat_col and num_cols:
        with col1:
            st.markdown(f"**{num_cols[0]} by {best_cat_col}**")
            fig_bar = px.bar(df, x=best_cat_col, y=num_cols[0], template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.markdown(f"**Distribution of {num_cols[0]}**")
            fig_pie = px.pie(df, names=best_cat_col, values=num_cols[0], template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
    elif not best_cat_col:
        st.info("No suitable categorical column found (needs 2-20 unique values) for Bar/Pie charts.")

    if date_cols and num_cols:
        st.markdown(f"**{num_cols[0]} Over Time**")
        df_grouped = df.groupby(date_cols[0])[num_cols[0]].sum().reset_index()
        fig_line = px.line(df_grouped, x=date_cols[0], y=num_cols[0], markers=True, template="plotly_dark")
        st.plotly_chart(fig_line, use_container_width=True)

    if len(num_cols) >= 2:
        st.markdown(f"**Correlation: {num_cols[0]} vs {num_cols[1]}**")
        fig_scat = px.scatter(df, x=num_cols[0], y=num_cols[1], template="plotly_dark")
        st.plotly_chart(fig_scat, use_container_width=True)

    st.divider()

    st.subheader("🤖 AI Executive Summary")
    if st.button("✨ Generate Insights"):
        if not GEMINI_CONFIGURED:
            st.error("Gemini AI is not configured.")
        else:
            with st.spinner("Analyzing data structure and statistics..."):
                try:
                    buffer = StringIO()
                    df.info(buf=buffer)
                    df_info = buffer.getvalue()
                    df_desc = df.describe().to_string()
                    prompt = f"""You are a senior data analyst. Provide 3-5 high-level bullet points on this data:
                    Data Info: {df_info}
                    Data Statistics: {df_desc}"""
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Analysis Failed: {e}")

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
        
        df_processed = st.session_state.df_processed
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizer", "🛠️ Data Explorer & Cleaner", "🚀 One-Click Dashboard", "🤖 ML Studio"])
        
        # === TAB 1: VISUALIZER ===
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

        # === TAB 2: EXPLORER & CLEANER ===
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
                
                # --- FIX: IMPROVED MISSING VALUE HANDLING ---
                with st.expander("💧 Handle Missing Values", expanded=True):
                    if missing_cells > 0:
                        st.dataframe(df_processed.isnull().sum()[df_processed.isnull().sum() > 0], height=100)
                        
                        # Split into different methods to handle text vs numbers correctly
                        method = st.selectbox("Method:", [
                            "Drop Rows with Missing Values", 
                            "Fill with Mean (Numeric Cols Only)", 
                            "Fill with Mode (Most Frequent - All Cols)", 
                            "Fill with Zero / 'Unknown'"
                        ])
                        
                        if st.button("Apply Fix"):
                            add_to_history(df_processed)
                            
                            if method == "Drop Rows with Missing Values":
                                st.session_state.df_processed = df_processed.dropna()
                                
                            elif method == "Fill with Mean (Numeric Cols Only)":
                                # Only fill numeric columns to prevent errors
                                num_cols = df_processed.select_dtypes(include=['number']).columns
                                st.session_state.df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].mean())
                                
                            elif method == "Fill with Mode (Most Frequent - All Cols)":
                                for col in df_processed.columns:
                                    if df_processed[col].isnull().any():
                                        st.session_state.df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                                        
                            elif method == "Fill with Zero / 'Unknown'":
                                num_cols = df_processed.select_dtypes(include=['number']).columns
                                cat_cols = df_processed.select_dtypes(exclude=['number']).columns
                                if len(num_cols) > 0:
                                    st.session_state.df_processed[num_cols] = df_processed[num_cols].fillna(0)
                                if len(cat_cols) > 0:
                                    st.session_state.df_processed[cat_cols] = df_processed[cat_cols].fillna("Unknown")
                                    
                            st.success("✅ Missing values handled!")
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
            # Use dynamic key to ensure refreshing when history changes
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

        # === TAB 4: ML STUDIO ===
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
