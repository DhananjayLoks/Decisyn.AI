import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import os
import google.generativeai as genai

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

# --- Caching Functions for Performance ---
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

# --- Helper function to save state for Undo ---
def add_to_history(df):
    st.session_state.history.append(df.copy())

# --- UI FUNCTION FOR CODE GENERATOR ---
def page_code_generator():
    # (Code generator function remains unchanged)
    st.header("Translate Natural Language to Code")
    st.markdown("Enter a description of what you want to do, and the AI will generate the Python code for it.")
    user_prompt = st.text_area("Your request:", height=150, placeholder="e.g., create a pandas dataframe with 5 rows of random data...")
    if st.button("Generate Code", type="primary"):
        if not GEMINI_CONFIGURED:
            st.error("Gemini AI is not configured. Please set the GOOGLE_API_KEY environment variable.")
            return
        if user_prompt:
            with st.spinner("🤖 Calling Gemini AI..."):
                try:
                    full_prompt = (
                        f"You are a senior staff software engineer specializing in clean, efficient, and idiomatic Python code. "
                        f"Your task is to generate a Python code snippet based on the user's request.\n\n"
                        f"Follow these rules strictly:\n"
                        f"1. Generate ONLY the raw Python code. Do not include explanations, comments, or markdown formatting like ```python.\n"
                        f"2. Write the most standard and conventional Python code possible.\n"
                        f"3. If a library is mentioned (e.g., NumPy), only use it if it's the most practical and conventional way to solve the problem. Do not force its use unnaturally.\n\n"
                        f"User's request: '{user_prompt}'"
                    )
                    response = model.generate_content(full_prompt)
                    st.code(response.text, language='python')
                except Exception as e:
                    st.error(f"An error occurred while generating the code: {e}")
        else:
            st.warning("Please enter a description.")

# --- UI FUNCTION FOR DATA ANALYZER ---
def page_data_analyzer():
    st.header("Analyze, Clean & Visualize Your Data")
    st.markdown("Upload a CSV or Excel file to get instant insights, clean data, and build interactive charts.")
    
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            st.error("Unsupported file format. Please upload a CSV or XLSX file.")
            return

        if 'df_processed' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
            st.session_state.df_processed = df.copy()
            st.session_state.file_name = uploaded_file.name
            st.session_state.history = [df.copy()]
        
        df_processed = st.session_state.df_processed
        
        tab1, tab2 = st.tabs(["📊 Visualizer", "📈 Data Explorer & Cleaner"])
        
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

                st.markdown("##### Customization")
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
                    st.error(f"Could not generate chart. Please check column compatibility. Error: {e}")

        with tab2:
            st.subheader("Data Explorer & Cleaner")
            
            # --- UNDO BUTTON ---
            if len(st.session_state.history) > 1:
                if st.button("↩️ Undo Last Action"):
                    st.session_state.history.pop()
                    st.session_state.df_processed = st.session_state.history[-1].copy()
                    st.success("✅ Last action undone.")
                    st.rerun()

            # --- AI-POWERED FEATURES ---
            with st.expander("💡 AI Cleaning Suggestions"):
                if st.button("Analyze for Suggestions"):
                    with st.spinner("🤖 Analyzing for suggestions..."):
                        try:
                            string_io = StringIO()
                            df_processed.info(buf=string_io)
                            df_info = string_io.getvalue()
                            df_head = df_processed.head().to_string()
                            prompt = f"""You are an expert data analyst. Analyze the dataset profile below and suggest 3-5 specific, actionable cleaning steps. Focus on issues like inconsistent data types, potential typos, inconsistent capitalization, or columns that could be transformed for better analysis.
                            **Dataframe Info:**\n{df_info}\n\n**First 5 Rows:**\n{df_head}"""
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"An error occurred while generating suggestions: {e}")
            
            with st.expander("🪄 Natural Language Cleaning (Experimental)"):
                st.warning("⚠️ This feature is experimental. Always review the changes to your data carefully.")
                nl_command = st.text_input("Enter a cleaning command in plain English:", placeholder="e.g., make the 'country' column lowercase")
                if st.button("Execute Command"):
                    with st.spinner("🤖 Translating and executing..."):
                        try:
                            prompt = f"""You are a Pandas expert. A user wants to modify their DataFrame, named `df_processed`. Their command is: '{nl_command}'. Generate ONLY the single line of Python code to perform this and reassign it back to `df_processed`. Example: for 'make country uppercase', output `df_processed['country'] = df_processed['country'].str.upper()`."""
                            code_response = model.generate_content(prompt)
                            generated_code = code_response.text.strip().replace("```python", "").replace("```", "")
                            
                            add_to_history(df_processed)
                            local_vars = {'df_processed': st.session_state.df_processed.copy(), 'pd': pd}
                            exec(generated_code, {}, local_vars)
                            st.session_state.df_processed = local_vars['df_processed']
                            st.success("✅ Command executed successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not execute command. Error: {e}")

            # --- CORE CLEANING TOOLKIT ---
            with st.expander("💧 Handle Missing Values"):
                missing_values = df_processed.isnull().sum()
                if missing_values.sum() > 0:
                    missing_df = missing_values[missing_values > 0].reset_index()
                    missing_df.columns = ['Column', 'Missing Count']
                    st.dataframe(missing_df)
                    fill_option = st.selectbox("Select method to handle missing values:", ('None', 'Drop rows with any missing value', 'Fill with Mean', 'Fill with Median', 'Fill with Mode', 'Fill with a specific value'))
                    fill_value = None
                    if fill_option == 'Fill with a specific value':
                        fill_value = st.text_input("Enter the value to fill with:")
                    if st.button("Apply Missing Value Action"):
                        add_to_history(df_processed)
                        df_cleaned = df_processed.copy()
                        if fill_option == 'Drop rows with any missing value': df_cleaned.dropna(inplace=True)
                        elif fill_option == 'Fill with Mean': df_cleaned.fillna(df_cleaned.select_dtypes(include=['number']).mean(), inplace=True)
                        elif fill_option == 'Fill with Median': df_cleaned.fillna(df_cleaned.select_dtypes(include=['number']).median(), inplace=True)
                        elif fill_option == 'Fill with Mode':
                            for col in df_cleaned.columns:
                                if df_cleaned[col].isnull().any(): df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                        elif fill_option == 'Fill with a specific value' and fill_value: df_cleaned.fillna(fill_value, inplace=True)
                        st.session_state.df_processed = df_cleaned
                        st.success("✅ Missing value action applied!")
                        st.rerun()
                else:
                    st.info("No missing values found.")
            
            with st.expander("🧹 Remove Duplicates"):
                duplicates_count = df_processed.duplicated().sum()
                st.write(f"Found **{duplicates_count}** duplicate rows.")
                if st.button("Remove Duplicate Rows", disabled=bool(duplicates_count == 0)):
                    add_to_history(df_processed)
                    st.session_state.df_processed = df_processed.drop_duplicates()
                    st.success("✅ Duplicates removed.")
                    st.rerun()

            with st.expander("🔄 Change Data Types"):
                col_to_change = st.selectbox("Select column to change type:", df_processed.columns, key="col_type")
                new_type = st.selectbox("Select new type:", ["object (text)", "int64 (integer)", "float64 (decimal)", "datetime64[ns] (date)"])
                if st.button("Apply Type Change"):
                    add_to_history(df_processed)
                    try:
                        if new_type.startswith("datetime"): st.session_state.df_processed[col_to_change] = pd.to_datetime(df_processed[col_to_change])
                        else: st.session_state.df_processed[col_to_change] = df_processed[col_to_change].astype(new_type.split(" ")[0])
                        st.success(f"✅ Changed type of '{col_to_change}' to {new_type}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not convert column. Error: {e}")

            with st.expander("✏️ Text Manipulation"):
                text_cols = df_processed.select_dtypes(include=['object']).columns
                col_to_text_manip = st.selectbox("Select text column:", text_cols, key="text_manip")
                manip_option = st.radio("Select operation:", ["Convert to Lowercase", "Convert to Uppercase", "Trim Whitespace"])
                if st.button("Apply Text Operation"):
                    add_to_history(df_processed)
                    if manip_option == "Convert to Lowercase": st.session_state.df_processed[col_to_text_manip] = df_processed[col_to_text_manip].str.lower()
                    elif manip_option == "Convert to Uppercase": st.session_state.df_processed[col_to_text_manip] = df_processed[col_to_text_manip].str.upper()
                    elif manip_option == "Trim Whitespace": st.session_state.df_processed[col_to_text_manip] = df_processed[col_to_text_manip].str.strip()
                    st.success(f"✅ Applied '{manip_option}' to '{col_to_text_manip}'.")
                    st.rerun()

            with st.expander("🔍 Find and Replace"):
                col_to_replace = st.selectbox("Select column:", df_processed.columns, key="find_replace")
                find_val = st.text_input("Find this value:")
                replace_val = st.text_input("Replace with this value:")
                if st.button("Apply Replace"):
                    if find_val:
                        add_to_history(df_processed)
                        st.session_state.df_processed[col_to_replace] = st.session_state.df_processed[col_to_replace].replace(find_val, replace_val)
                        st.success(f"✅ Replaced '{find_val}' with '{replace_val}' in '{col_to_replace}'.")
                        st.rerun()
                    else:
                        st.warning("Please enter a value to find.")

            st.markdown("---")
            st.subheader("Current Data")
            st.dataframe(df_processed)
            st.markdown("---")
            st.subheader("Download Processed Data")
            st.download_button(label="📥 Download CSV", data=convert_df_to_csv(df_processed), file_name=f"processed_{uploaded_file.name}.csv", mime="text/csv")
            
            st.markdown("---")
            st.subheader("🤖 AI-Powered Data Summary")
            if st.button("Generate Summary"):
                with st.spinner("🧠 Analyzing your data..."):
                    # AI Summary logic...
                    pass

    else:
        st.info("Please upload a CSV or Excel file to begin analysis.")
        if 'df_processed' in st.session_state:
            del st.session_state.df_processed
            del st.session_state.history

# --- SIDEBAR AND MAIN APP LOGIC ---
with st.sidebar:
    st.title("Decisyn AI")
    st.image("logo.jpg", width=120)
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