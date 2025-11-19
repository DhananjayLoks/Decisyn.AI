import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import os
import google.generativeai as genai
import matplotlib.pyplot as plt # For Dendrograms

# --- ML Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as shc # For Hierarchical Clustering

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
# 🤖 PRO ML STUDIO FUNCTION
# ==========================================
def render_ml_studio(df):
    st.markdown("## 🤖 Pro ML Studio")
    st.markdown("Advanced analytics, forecasting, and segmentation.")

    # Tabs for different ML Tasks
    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["🔮 Forecast Future", "🎯 Target Prediction (Supervised)", "🧬 Advanced Clustering"])

    # === TAB 1: TIME SERIES FORECASTING ===
    with ml_tab1:
        st.subheader("Time-Series Forecasting")
        st.caption("Predict future values based on historical trends.")

        date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()

        if not date_cols:
            st.error("⚠️ No Date column found! Please convert a column to 'datetime' in the Cleaner tab first.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                date_col = st.selectbox("Select Date Column:", date_cols)
            with c2:
                target_col = st.selectbox("Select Value to Predict (e.g., Sales):", num_cols)
            with c3:
                horizon = st.slider("Forecast Horizon (Years):", 1, 10, 5)

            if st.button("🚀 Generate Forecast"):
                # 1. Prepare Data (Group by date to handle duplicates)
                df_grouped = df.groupby(date_col)[target_col].sum().reset_index()
                df_grouped = df_grouped.sort_values(date_col)
                
                # 2. Create Time Features (Ordinal dates for Regression)
                df_grouped['Time_Index'] = np.arange(len(df_grouped))
                
                # 3. Train Linear Trend Model
                X = df_grouped[['Time_Index']]
                y = df_grouped[target_col]
                model = LinearRegression()
                model.fit(X, y)
                
                # 4. Create Future DataFrame
                last_date = df_grouped[date_col].max()
                # Assuming monthly data roughly; extending index
                future_steps = horizon * 12 # 12 months per year estimate
                future_indices = np.arange(len(df_grouped), len(df_grouped) + future_steps).reshape(-1, 1)
                future_preds = model.predict(future_indices)
                
                # Create Future Dates (Approximate)
                future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq='M')[1:]
                
                df_future = pd.DataFrame({
                    date_col: future_dates,
                    target_col: future_preds,
                    'Type': 'Forecast'
                })
                
                df_grouped['Type'] = 'History'
                
                # Combine
                df_final = pd.concat([df_grouped[[date_col, target_col, 'Type']], df_future])
                
                # 5. Visualization
                fig = px.line(df_final, x=date_col, y=target_col, color='Type', 
                              title=f"{target_col} Forecast: Next {horizon} Years",
                              color_discrete_map={"History": "cyan", "Forecast": "orange"},
                              template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Forecast generated for {horizon} years into the future!")

    # === TAB 2: SUPERVISED LEARNING (REGRESSION/CLASSIFICATION) ===
    with ml_tab2:
        st.subheader("Predictive Modeling & What-If Analysis")
        
        model_type = st.selectbox("Choose Algorithm:", ["Linear Regression (Predict Numbers)", "Logistic Regression (Predict Categories)"])
        
        c1, c2 = st.columns(2)
        with c1: target = st.selectbox("Target Variable (y):", df.columns)
        with c2: features = st.multiselect("Feature Variables (X):", [c for c in df.columns if c != target])
        
        if st.button("Train & Build Calculator") and features:
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            # Encoding
            X = pd.get_dummies(X, drop_first=True)
            le = None
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = None
            if model_type.startswith("Linear"):
                model = LinearRegression()
                model.fit(X_train, y_train)
                score = r2_score(y_test, model.predict(X_test))
                st.metric("Model Accuracy (R²)", f"{score:.2%}")
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                st.metric("Model Accuracy", f"{score:.2%}")
                
            st.markdown("---")
            st.subheader("🧮 What-If Calculator")
            st.markdown("Change the inputs below to predict the outcome.")
            
            # Dynamic Input Generation
            user_inputs = {}
            cols = st.columns(len(features))
            for i, col_name in enumerate(features):
                with cols[i % len(features)]:
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        val = st.number_input(f"{col_name}", value=float(df[col_name].mean()))
                    else:
                        val = st.selectbox(f"{col_name}", df[col_name].unique())
                    user_inputs[col_name] = val
            
            if st.button("Predict Outcome"):
                # Prepare input data matching training shape
                input_df = pd.DataFrame([user_inputs])
                input_df = pd.get_dummies(input_df)
                # Align columns (missing columns get 0)
                input_df = input_df.reindex(columns=X.columns, fill_value=0)
                
                prediction = model.predict(input_df)[0]
                
                if le: # Decode if categorical
                    prediction = le.inverse_transform([prediction])[0]
                    
                st.success(f"### 🔮 Predicted {target}: {prediction}")

    # === TAB 3: ADVANCED CLUSTERING ===
    with ml_tab3:
        st.subheader("Advanced Segmentation")
        cluster_method = st.radio("Method:", ["K-Means (Group & Visualize)", "Hierarchical (Dendrogram)"], horizontal=True)
        
        cluster_cols = st.multiselect("Select Features to Cluster:", df.select_dtypes(include=['number']).columns)
        
        if cluster_cols:
            X = df[cluster_cols].dropna()
            
            if cluster_method.startswith("K-Means"):
                k = st.slider("Number of Clusters (k):", 2, 8, 3)
                if st.button("Run K-Means"):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    df['Cluster'] = kmeans.fit_predict(X).astype(str)
                    
                    # Advanced 3D or 2D Plot
                    if len(cluster_cols) >= 3:
                        fig = px.scatter_3d(df, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2], color='Cluster', title="3D Cluster Visualization", template="plotly_dark")
                    elif len(cluster_cols) == 2:
                        fig = px.scatter(df, x=cluster_cols[0], y=cluster_cols[1], color='Cluster', title="2D Cluster Visualization", template="plotly_dark")
                        # Add Centroids
                        centroids = kmeans.cluster_centers_
                        fig.add_trace(go.Scatter(x=centroids[:,0], y=centroids[:,1], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Centroids'))
                    else:
                        fig = px.scatter(df, x=df.index, y=cluster_cols[0], color='Cluster', title="1D Cluster Visualization", template="plotly_dark")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            elif cluster_method.startswith("Hierarchical"):
                st.subheader("🌳 Dendrogram Visualization")
                st.caption("This chart helps you decide how many clusters exist naturally in your data.")
                
                if st.button("Generate Dendrogram"):
                    with st.spinner("Calculating Linkages... (This may take a moment)"):
                        # limit rows for performance if huge
                        if len(X) > 2000:
                            st.warning("Dataset too large for clean Dendrogram. Sampling first 1000 rows.")
                            X_sample = X.head(1000)
                        else:
                            X_sample = X
                            
                        plt.figure(figsize=(10, 7))
                        plt.title("Customer Dendrogram")
                        dend = shc.dendrogram(shc.linkage(X_sample, method='ward'))
                        plt.axhline(y=6, color='r', linestyle='--')
                        st.pyplot(plt)


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
