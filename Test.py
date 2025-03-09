import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# Page configuration
st.set_page_config(page_title="Data Analyzer", page_icon="📊", layout="wide")

# Get API key from secrets.toml if available
api_key = ""
if "gemini_api_key" in st.secrets:
    api_key = st.secrets["gemini_api_key"]

# Function to load default data
@st.cache_data
def load_default_data():
    df = pd.read_csv('data.csv')
    # Remove any comment rows that might exist in the CSV
    df = df[~df['Year'].astype(str).str.contains('//')]
    return df

# Replace the existing load_data function with this more flexible version
def load_data():
    # Check if data is already in session state
    if 'data' not in st.session_state:
        # Initialize with default data
        st.session_state.data = load_default_data()
        st.session_state.using_default = True
        st.session_state.data_source = "Default New Zealand industry dataset"
    
    # Add data upload option in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Data Options")
    
    # Option to upload custom data
    uploaded_file = st.sidebar.file_uploader("Upload your own dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Try to load the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Update session state with new data
            st.session_state.data = df
            st.session_state.using_default = False
            st.session_state.data_source = uploaded_file.name
            st.sidebar.success(f"Custom dataset '{uploaded_file.name}' loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            st.sidebar.info("Using default dataset instead.")
    
    # Add button to reset to default data
    if not st.session_state.using_default:
        if st.sidebar.button("Reset to Default Dataset"):
            st.session_state.data = load_default_data()
            st.session_state.using_default = True
            st.session_state.data_source = "Default New Zealand industry dataset"
            st.sidebar.success("Reset to default dataset!")
    else:
        st.sidebar.info("Currently using default dataset.")
    
    return st.session_state.data

# Main function to run the app
def main():
    st.title("Data Analyzer")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Data Explorer", "Charts", "AI Assistant"]
    selection = st.sidebar.radio("Go to", pages)
    
    # Load data (with upload option)
    df = load_data()
    
    if selection == "Data Explorer":
        data_explorer(df)
    elif selection == "Charts":
        charts(df)
    else:
        ai_assistant(df, api_key)

def data_explorer(df):
    st.header("Data Explorer")
    
    # Display basic statistics
    st.subheader("Dataset Overview")
    st.write(f"Data source: {st.session_state.data_source}")
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Show column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Values': df.count().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info)
    
    # Allow user to filter data by available columns
    st.subheader("Data Filter")
    
    # Dynamically determine columns suitable for filtering (categorical or low cardinality)
    filter_columns = []
    for col in df.columns:
        if df[col].nunique() < 50 or df[col].dtype == 'object':
            filter_columns.append(col)
    
    # Display up to 3 filter columns side by side
    filter_cols = st.columns(min(3, max(1, len(filter_columns))))
    
    selected_filters = {}
    for i, col in enumerate(filter_columns[:min(3, len(filter_columns))]):
        with filter_cols[i % 3]:
            unique_values = sorted(df[col].unique())
            selected_filters[col] = st.selectbox(f"Select {col}", ["All"] + list(unique_values))
    
    # Filter data based on selection
    filtered_df = df.copy()
    for col, value in selected_filters.items():
        if value != "All":
            filtered_df = filtered_df[filtered_df[col] == value]
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

def charts(df):
    st.header("Data Visualization")
    
    # Check if dataframe is empty
    if df.empty:
        st.warning("The dataset is empty. Please upload a valid dataset.")
        return
    
    # Determine if we're using the default dataset or a custom one
    using_default_data = st.session_state.get('using_default', True)
    
    if using_default_data:
        # Original visualization options for the default New Zealand industry data
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Industry Comparison", "Time Series", "Financial Position Analysis", "Performance Indicators"]
        )
        
        if viz_type == "Industry Comparison":
            industry_comparison(df)
        elif viz_type == "Time Series":
            time_series(df)
        elif viz_type == "Financial Position Analysis":
            financial_position(df)
        else:
            performance_indicators(df)
    else:
        # Generic visualization options for any uploaded dataset
        dynamic_visualizations(df)

def dynamic_visualizations(df):
    """Provides visualization options that adapt to any dataset structure"""
    
    # Determine numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset. Cannot create visualizations.")
        return
    
    # Choose visualization type based on available data
    viz_options = ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Heatmap"]
    viz_type = st.selectbox("Select Visualization Type", viz_options)
    
    if viz_type == "Bar Chart":
        create_bar_chart(df, categorical_cols, numeric_cols)
    
    elif viz_type == "Line Chart":
        create_line_chart(df, categorical_cols, numeric_cols)
        
    elif viz_type == "Scatter Plot":
        create_scatter_plot(df, numeric_cols)
        
    elif viz_type == "Pie Chart":
        create_pie_chart(df, categorical_cols, numeric_cols)
        
    elif viz_type == "Histogram":
        create_histogram(df, numeric_cols)
        
    elif viz_type == "Box Plot":
        create_box_plot(df, categorical_cols, numeric_cols)
        
    elif viz_type == "Heatmap":
        create_heatmap(df, numeric_cols)

def create_bar_chart(df, categorical_cols, numeric_cols):
    st.subheader("Bar Chart")
    
    if not categorical_cols:
        st.warning("No categorical columns found for x-axis. Bar chart requires categorical data.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis (categorical):", categorical_cols)
    with col2:
        y_col = st.selectbox("Select Y-axis (numeric):", numeric_cols)
    
    # Optional color grouping
    color_col = st.selectbox("Group by color (optional):", ["None"] + categorical_cols)
    color_col = None if color_col == "None" else color_col
    
    # Limit categories if too many
    value_counts = df[x_col].value_counts()
    if len(value_counts) > 15:
        st.warning(f"Too many categories ({len(value_counts)}) for {x_col}. Showing top 15.")
        top_cats = value_counts.nlargest(15).index.tolist()
        chart_df = df[df[x_col].isin(top_cats)]
    else:
        chart_df = df
    
    # Create chart
    fig = px.bar(
        chart_df, 
        x=x_col, 
        y=y_col,
        color=color_col,
        title=f"{y_col} by {x_col}",
        labels={x_col: x_col, y_col: y_col},
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_line_chart(df, categorical_cols, numeric_cols):
    st.subheader("Line Chart")
    
    col1, col2 = st.columns(2)
    with col1:
        x_options = categorical_cols + numeric_cols
        x_col = st.selectbox("Select X-axis:", x_options)
    with col2:
        y_col = st.selectbox("Select Y-axis (numeric):", numeric_cols)
    
    # Optional color grouping
    color_col = st.selectbox("Group by color (optional):", ["None"] + categorical_cols)
    color_col = None if color_col == "None" else color_col
    
    # Create chart
    if x_col in categorical_cols and len(df[x_col].unique()) > 30:
        st.warning(f"Too many categories ({len(df[x_col].unique())}) for {x_col}. Consider a different visualization.")
        return
    
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=f"{y_col} by {x_col}",
        labels={x_col: x_col, y_col: y_col},
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_scatter_plot(df, numeric_cols):
    st.subheader("Scatter Plot")
    
    if len(numeric_cols) < 2:
        st.warning("Scatter plot requires at least 2 numeric columns.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("Select X-axis:", numeric_cols)
    with col2:
        y_col = st.selectbox("Select Y-axis:", [col for col in numeric_cols if col != x_col] if len(numeric_cols) > 1 else numeric_cols)
    with col3:
        size_col = st.selectbox("Size by (optional):", ["None"] + numeric_cols)
        size_col = None if size_col == "None" else size_col
    
    # Optional color grouping (categorical)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
    color_col = None if color_col == "None" else color_col
    
    # Create chart
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        size=size_col,
        color=color_col,
        title=f"{y_col} vs {x_col}",
        labels={x_col: x_col, y_col: y_col},
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_pie_chart(df, categorical_cols, numeric_cols):
    st.subheader("Pie Chart")
    
    if not categorical_cols:
        st.warning("Pie chart requires at least one categorical column.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        names_col = st.selectbox("Select category column:", categorical_cols)
    with col2:
        values_col = st.selectbox("Select values column:", numeric_cols)
    
    # Limit categories if too many
    value_counts = df[names_col].value_counts()
    if len(value_counts) > 10:
        st.warning(f"Too many categories ({len(value_counts)}) for {names_col}. Showing top 10.")
        top_cats = value_counts.nlargest(10).index.tolist()
        chart_df = df[df[names_col].isin(top_cats)].copy()
        
        # Sum the rest as "Others"
        others_sum = df[~df[names_col].isin(top_cats)][values_col].sum()
        if others_sum > 0:
            others_df = pd.DataFrame({names_col: ["Others"], values_col: [others_sum]})
            chart_df = pd.concat([chart_df, others_df])
    else:
        chart_df = df
    
    # Group by the selected category and sum the values
    pie_data = chart_df.groupby(names_col)[values_col].sum().reset_index()
    
    # Create chart
    fig = px.pie(
        pie_data, 
        names=names_col, 
        values=values_col,
        title=f"Distribution of {values_col} by {names_col}",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_histogram(df, numeric_cols):
    st.subheader("Histogram")
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select column for histogram:", numeric_cols)
    with col2:
        bins = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
    
    # Optional color grouping
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
    color_col = None if color_col == "None" else color_col
    
    # Create chart
    fig = px.histogram(
        df, 
        x=x_col,
        color=color_col,
        nbins=bins,
        title=f"Distribution of {x_col}",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_box_plot(df, categorical_cols, numeric_cols):
    st.subheader("Box Plot")
    
    col1, col2 = st.columns(2)
    with col1:
        y_col = st.selectbox("Select numeric column (y-axis):", numeric_cols)
    with col2:
        x_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
        x_col = None if x_col == "None" else x_col
    
    # Create chart
    if x_col is not None and len(df[x_col].unique()) > 15:
        st.warning(f"Too many categories ({len(df[x_col].unique())}) for {x_col}. Showing top 15.")
        top_cats = df[x_col].value_counts().nlargest(15).index.tolist()
        chart_df = df[df[x_col].isin(top_cats)]
    else:
        chart_df = df
    
    fig = px.box(
        chart_df, 
        x=x_col, 
        y=y_col,
        title=f"Box plot of {y_col}" + (f" by {x_col}" if x_col else ""),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_heatmap(df, numeric_cols):
    st.subheader("Correlation Heatmap")
    
    if len(numeric_cols) < 2:
        st.warning("Heatmap requires at least 2 numeric columns.")
        return
    
    # Allow user to select columns for correlation
    selected_cols = st.multiselect(
        "Select columns for correlation matrix:", 
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
    
    if len(selected_cols) < 2:
        st.warning("Please select at least 2 columns.")
        return
    
    # Calculate correlation matrix
    corr = df[selected_cols].corr()
    
    # Create heatmap with Plotly
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
        height=600,
        width=700
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Modify AI Assistant for generic dataset handling
def ai_assistant(df, api_key):
    st.header("AI Data Assistant")
    
    if not api_key:
        st.warning("No Gemini API key found in secrets.toml. Some features may not work.")
        st.info("To enable AI features, add your Gemini API key to the .streamlit/secrets.toml file.")
    
    # User query input
    user_query = st.text_input("Ask a question about the data:", 
                               placeholder="e.g., What insights can I draw from this dataset?")
    
    if st.button("Get Answer") and user_query:
        with st.spinner("Analyzing data..."):
            try:
                if api_key:
                    # If we have an API key, use Gemini to generate response
                    import google.generativeai as genai
                    
                    # Configure the API
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    # Generate data summary to help the AI understand the dataset
                    data_summary = (
                        f"Dataset overview:\n"
                        f"- Data source: {st.session_state.get('data_source', 'Unknown')}\n"
                        f"- Number of records: {df.shape[0]}\n"
                        f"- Number of columns: {df.shape[1]}\n"
                        f"- Column names: {', '.join(df.columns)}\n"
                        f"- Numeric columns: {', '.join(df.select_dtypes(include=['int64', 'float64']).columns)}\n"
                        f"- Categorical columns: {', '.join(df.select_dtypes(include=['object', 'category']).columns)}\n"
                        f"- First few rows:\n{df.head(3).to_string()}\n"
                    )
                    
                    # Additional dataset statistics
                    stats_summary = ""
                    try:
                        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                        if not numeric_cols.empty:
                            stats_summary = f"Numeric column statistics:\n{df[numeric_cols].describe().to_string()}\n\n"
                    except Exception as e:
                        stats_summary = f"Could not generate statistics: {str(e)}\n\n"
                    
                    # Prepare context for the AI
                    context = f"""
                    You are an expert data analyst examining a dataset.
                    Please analyze this data and answer the user's question.
                    
                    {data_summary}
                    
                    {stats_summary}
                    
                    Please provide a clear, concise answer to the user's question.
                    Include relevant statistics, trends, or insights from the data.
                    If you're not sure about something, acknowledge the limitations of the data.
                    """
                    
                    # Send query to Gemini
                    response = model.generate_content(context + f"\n\nQuestion: {user_query}")
                    
                    # Display response
                    st.success("Response generated!")
                    st.write(response.text)
                    
                    # Create a relevant visualization based on the dataset
                    st.subheader("Data Visualization")
                    
                    # Determine what kind of visualization would be most helpful
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if numeric_cols and categorical_cols:
                        # Show a bar chart of the first categorical vs. first numeric column
                        cat_col = categorical_cols[0]
                        num_col = numeric_cols[0]
                        
                        # Limit categories if too many
                        if df[cat_col].nunique() > 15:
                            top_cats = df[cat_col].value_counts().nlargest(15).index
                            chart_df = df[df[cat_col].isin(top_cats)]
                        else:
                            chart_df = df
                        
                        fig = px.bar(
                            chart_df, 
                            x=cat_col, 
                            y=num_col,
                            title=f"{num_col} by {cat_col}",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif len(numeric_cols) >= 2:
                        # Show a scatter plot of the first two numeric columns
                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1]
                        
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            title=f"{y_col} vs {x_col}",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif numeric_cols:
                        # Show a histogram of the first numeric column
                        num_col = numeric_cols[0]
                        
                        fig = px.histogram(
                            df, 
                            x=num_col,
                            title=f"Distribution of {num_col}",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif categorical_cols:
                        # Show a pie chart of the first categorical column
                        cat_col = categorical_cols[0]
                        
                        # Limit categories if too many
                        if df[cat_col].nunique() > 10:
                            value_counts = df[cat_col].value_counts().nlargest(10)
                            others = pd.Series({'Others': df[cat_col].value_counts().nsmallest(df[cat_col].nunique() - 10).sum()})
                            counts = pd.concat([value_counts, others])
                        else:
                            counts = df[cat_col].value_counts()
                        
                        fig = px.pie(
                            values=counts.values,
                            names=counts.index,
                            title=f"Distribution of {cat_col}",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("For AI-powered analysis, please add a Gemini API key to your secrets.toml file.")
                    
                    # Show a basic summary of the data
                    st.subheader("Dataset Summary")
                    
                    # Show column types
                    col_types = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null Values': df.count().values,
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_types)
                    
                    # Show a basic visualization
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if numeric_cols:
                        st.subheader("Sample Visualization")
                        num_col = numeric_cols[0]
                        
                        fig = px.histogram(
                            df, 
                            x=num_col,
                            title=f"Distribution of {num_col}",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please try a different question or check your data format.")

if __name__ == "__main__":
    main()