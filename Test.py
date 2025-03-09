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
st.set_page_config(page_title="Industry Data Analyzer", page_icon="📊", layout="wide")

# Get API key from secrets.toml if available
api_key = ""
if "gemini_api_key" in st.secrets:
    api_key = st.secrets["gemini_api_key"]

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    # Remove any comment rows that might exist in the CSV
    df = df[~df['Year'].astype(str).str.contains('//')]
    return df

# Main function to run the app
def main():
    st.title("New Zealand Industry Data Analyzer")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Data Explorer", "Charts", "AI Assistant"]
    selection = st.sidebar.radio("Go to", pages)
    
    # Load data
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
    st.write(f"Number of records: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Allow user to filter by year, industry, and variable name
    col1, col2, col3 = st.columns(3)
    
    with col1:
        years = df['Year'].unique()
        selected_year = st.selectbox("Select Year", years)
    
    with col2:
        industries = sorted(df['Industry_name_NZSIOC'].unique())
        selected_industry = st.selectbox("Select Industry", industries)
    
    with col3:
        variables = sorted(df['Variable_name'].unique())
        selected_variable = st.selectbox("Select Variable", variables)
    
    # Filter data based on selection
    filtered_df = df[(df['Year'] == selected_year) & 
                    (df['Industry_name_NZSIOC'] == selected_industry) & 
                    (df['Variable_name'] == selected_variable)]
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

def charts(df):
    st.header("Data Visualization")
    
    # Choose visualization type
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

def industry_comparison(df):
    st.subheader("Industry Financial Comparison")
    
    # Filter to specific categories and metrics
    year = 2023  # Using the year from your dataset
    variables = ["Total income", "Total expenditure"]
    
    # Check if these variables exist in the data
    available_vars = [var for var in variables if var in df['Variable_name'].unique()]
    
    if not available_vars:
        st.warning("No matching variables found in the dataset.")
        # Add all variables for selection
        available_vars = df['Variable_name'].unique()
        selected_vars = st.multiselect("Select financial metrics to compare:", 
                                      options=available_vars,
                                      default=available_vars[:2] if len(available_vars) > 1 else available_vars)
    else:
        selected_vars = st.multiselect("Select financial metrics to compare:", 
                                      options=df['Variable_name'].unique(),
                                      default=available_vars)
    
    if not selected_vars:
        st.warning("Please select at least one variable to visualize.")
        return
    
    # Get industries by aggregation level
    agg_levels = df['Industry_aggregation_NZSIOC'].unique()
    selected_agg = st.selectbox("Select Industry Aggregation Level:", agg_levels)
    
    # Filter data for visualization
    viz_df = df[(df['Year'] == year) & 
                (df['Industry_aggregation_NZSIOC'] == selected_agg) & 
                (df['Variable_name'].isin(selected_vars))]
    
    if viz_df.empty:
        st.warning("No data available for the selected criteria.")
        return
    
    # Create a bar chart using Plotly
    fig = px.bar(
        viz_df, 
        x="Industry_name_NZSIOC", 
        y="Value", 
        color="Variable_name",
        title=f"Financial Comparison Across Industries ({year})",
        labels={"Industry_name_NZSIOC": "Industry", "Value": "Amount (millions)"},
        barmode="group"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def time_series(df):
    st.subheader("Time Series Analysis")
    
    # Check if we have multiple years
    years = sorted(df['Year'].unique())
    
    if len(years) <= 1:
        st.info("Note: This sample data only contains data for a single year. Time series visualization would require multiple years.")
        
        # Instead, let's visualize industry comparison for different metrics
        st.subheader("Industry Comparison")
        variables = sorted(df['Variable_name'].unique())
        selected_variable = st.selectbox("Select Financial Metric:", variables)
        
        # Get industries by aggregation level
        agg_levels = df['Industry_aggregation_NZSIOC'].unique()
        selected_agg = st.selectbox("Select Industry Aggregation Level:", agg_levels)
        
        # Filter data for visualization
        viz_df = df[(df['Industry_aggregation_NZSIOC'] == selected_agg) & 
                    (df['Variable_name'] == selected_variable)]
        
        if viz_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        fig = px.bar(
            viz_df,
            x="Industry_name_NZSIOC",
            y="Value",
            title=f"{selected_variable} by Industry ({years[0]})",
            labels={"Industry_name_NZSIOC": "Industry", "Value": "Amount (millions)"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # If we have multiple years, create a time series plot
        st.info("Multiple years of data available. Creating time series visualization.")
        
        # Select industry and variable
        industries = sorted(df['Industry_name_NZSIOC'].unique())
        selected_industry = st.selectbox("Select Industry:", industries)
        
        variables = sorted(df['Variable_name'].unique())
        selected_variable = st.selectbox("Select Financial Metric:", variables)
        
        # Filter data for visualization
        viz_df = df[(df['Industry_name_NZSIOC'] == selected_industry) & 
                   (df['Variable_name'] == selected_variable)]
        
        if viz_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        fig = px.line(
            viz_df,
            x="Year",
            y="Value",
            title=f"{selected_variable} for {selected_industry} Over Time",
            labels={"Year": "Year", "Value": "Amount (millions)"}
        )
        
        st.plotly_chart(fig, use_container_width=True)

def financial_position(df):
    st.subheader("Financial Position Analysis")
    
    # Since we don't have 'Level' column, let's use Industry_aggregation_NZSIOC
    agg_levels = df['Industry_aggregation_NZSIOC'].unique()
    selected_agg = st.selectbox("Select Industry Aggregation Level:", agg_levels)
    
    # Position-related variables that might be in the data
    position_vars = ["Total assets", "Current assets", "Fixed tangible assets", 
                     "Total equity and liabilities", "Current liabilities"]
    
    # Check which variables exist in the dataset
    available_position_vars = [var for var in position_vars if var in df['Variable_name'].unique()]
    
    if not available_position_vars:
        st.warning("No financial position variables found in the dataset.")
        st.info("Available variables: " + ", ".join(sorted(df['Variable_name'].unique())[:10]) + "...")
        
        # Let user select from available variables
        selected_vars = st.multiselect("Select financial variables to analyze:", 
                                      options=sorted(df['Variable_name'].unique()),
                                      default=sorted(df['Variable_name'].unique())[:3] if len(df['Variable_name'].unique()) > 2 else df['Variable_name'].unique())
        
        if not selected_vars:
            return
            
        # Select an industry to analyze
        industries = sorted(df[df['Industry_aggregation_NZSIOC'] == selected_agg]['Industry_name_NZSIOC'].unique())
        if not industries:
            st.warning("No industries found for the selected aggregation level.")
            return
            
        selected_ind = st.selectbox("Select Industry to Analyze:", industries)
        
        # Filter for visualization
        ind_df = df[(df['Industry_name_NZSIOC'] == selected_ind) & 
                    (df['Variable_name'].isin(selected_vars))]
        
        if ind_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        # Create a pie chart
        fig = px.pie(
            ind_df, 
            names="Variable_name", 
            values="Value",
            title=f"Financial Breakdown - {selected_ind}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show the data in a table
        st.subheader("Financial Data")
        st.dataframe(ind_df[['Variable_name', 'Value', 'Units']])
    else:
        # We have some position variables, proceed with visualization
        st.success(f"Found {len(available_position_vars)} financial position variables in the dataset.")
        
        # Select an industry to analyze
        industries = sorted(df[df['Industry_aggregation_NZSIOC'] == selected_agg]['Industry_name_NZSIOC'].unique())
        if not industries:
            st.warning("No industries found for the selected aggregation level.")
            return
            
        selected_ind = st.selectbox("Select Industry to Analyze:", industries)
        
        # Filter for selected industry and variables
        ind_df = df[(df['Industry_name_NZSIOC'] == selected_ind) & 
                    (df['Variable_name'].isin(available_position_vars))]
        
        if ind_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        # Create a pie chart
        fig = px.pie(
            ind_df, 
            names="Variable_name", 
            values="Value",
            title=f"Financial Position - {selected_ind}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show the data in a table
        st.subheader("Financial Position Data")
        st.dataframe(ind_df[['Variable_name', 'Value', 'Units']])

def performance_indicators(df):
    st.subheader("Performance Indicators")
    
    # Check for financial ratios in the dataset
    financial_ratios = df[df['Variable_category'] == 'Financial ratios']
    
    if financial_ratios.empty:
        st.info("No financial ratios found in the dataset. Showing other performance metrics.")
        
        # Show a selection of metrics instead
        performance_categories = df['Variable_category'].unique()
        selected_category = st.selectbox("Select Variable Category:", performance_categories)
        
        # Filter variables by category
        category_vars = sorted(df[df['Variable_category'] == selected_category]['Variable_name'].unique())
        selected_var = st.selectbox("Select Performance Metric:", category_vars)
        
        # Get industries by aggregation level
        agg_levels = df['Industry_aggregation_NZSIOC'].unique()
        selected_agg = st.selectbox("Select Industry Aggregation Level:", agg_levels)
        
        # Filter data for visualization
        viz_df = df[(df['Variable_category'] == selected_category) & 
                   (df['Variable_name'] == selected_var) & 
                   (df['Industry_aggregation_NZSIOC'] == selected_agg)]
        
        if viz_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        # Create a horizontal bar chart
        fig = px.bar(
            viz_df,
            y="Industry_name_NZSIOC",
            x="Value",
            title=f"{selected_var} by Industry",
            labels={"Industry_name_NZSIOC": "Industry", "Value": viz_df['Units'].iloc[0]},
            orientation='h'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # We have financial ratios, proceed with ratio visualization
        ratio_names = sorted(financial_ratios['Variable_name'].unique())
        selected_ratio = st.selectbox("Select Performance Ratio:", ratio_names)
        
        # Get industries by aggregation level
        agg_levels = df['Industry_aggregation_NZSIOC'].unique()
        selected_agg = st.selectbox("Select Industry Aggregation Level:", agg_levels)
        
        # Filter data for visualization
        ratio_df = financial_ratios[(financial_ratios['Variable_name'] == selected_ratio) & 
                                   (financial_ratios['Industry_aggregation_NZSIOC'] == selected_agg)]
        
        if ratio_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        # Create a horizontal bar chart
        fig = px.bar(
            ratio_df,
            y="Industry_name_NZSIOC",
            x="Value",
            title=f"{selected_ratio} by Industry",
            labels={"Industry_name_NZSIOC": "Industry", "Value": ratio_df['Units'].iloc[0]},
            orientation='h'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def ai_assistant(df, api_key):
    st.header("AI Data Assistant")
    
    if not api_key:
        st.warning("No Gemini API key found in secrets.toml. Some features may not work.")
        st.info("To enable AI features, add your Gemini API key to the .streamlit/secrets.toml file.")
    
    # User query input
    user_query = st.text_input("Ask a question about the New Zealand industry data:", 
                               placeholder="e.g., What industry had the highest total income in 2023?")
    
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
                        f"- Number of records: {df.shape[0]}\n"
                        f"- Years covered: {', '.join(map(str, sorted(df['Year'].unique())))}\n"
                        f"- Industry aggregation levels: {', '.join(sorted(df['Industry_aggregation_NZSIOC'].unique()))}\n"
                        f"- Number of unique industries: {df['Industry_name_NZSIOC'].nunique()}\n"
                        f"- Variable categories: {', '.join(sorted(df['Variable_category'].unique()))}\n"
                        f"- Top 10 variables: {', '.join(sorted(df['Variable_name'].unique())[:10])}\n"
                    )
                    
                    # Prepare sample data to help the AI understand the structure
                    sample_data = df.head(5).to_string()
                    
                    # Prepare context for the AI
                    context = f"""
                    You are an expert data analyst examining New Zealand industry financial data.
                    You are analyzing data for the year(s): {', '.join(map(str, sorted(df['Year'].unique())))}.
                    
                    {data_summary}
                    
                    Sample data format:
                    {sample_data}
                    
                    Please provide a clear, concise answer to the user's question.
                    Include relevant statistics, trends, or comparisons.
                    If you're not sure about something, acknowledge the limitations of the data.
                    """
                    
                    # Send query to Gemini
                    response = model.generate_content(context + f"\n\nQuestion: {user_query}")
                    
                    # Display response
                    st.success("Response generated!")
                    st.write(response.text)
                    
                    # Attempt to create a relevant visualization based on the query
                    st.subheader("Related Visualization")
                    
                    # Simple logic to detect query intent and show relevant chart
                    query_lower = user_query.lower()
                    if "income" in query_lower or "revenue" in query_lower:
                        # Show income comparison
                        income_df = df[df['Variable_name'] == 'Total income']
                        if not income_df.empty:
                            fig = px.bar(
                                income_df, 
                                x="Industry_name_NZSIOC", 
                                y="Value", 
                                title="Total Income by Industry",
                                labels={"Industry_name_NZSIOC": "Industry", "Value": "Income (millions)"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No income data available for visualization.")
                    elif "expenditure" in query_lower or "expense" in query_lower:
                        # Show expenditure comparison
                        expenditure_df = df[df['Variable_name'] == 'Total expenditure']
                        if not expenditure_df.empty:
                            fig = px.bar(
                                expenditure_df, 
                                x="Industry_name_NZSIOC", 
                                y="Value", 
                                title="Total Expenditure by Industry",
                                labels={"Industry_name_NZSIOC": "Industry", "Value": "Expenditure (millions)"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No expenditure data available for visualization.")
                    else:
                        # Show a summary of variable categories
                        var_summary = df.groupby('Variable_category').size().reset_index(name='Count')
                        fig = px.pie(
                            var_summary,
                            names='Variable_category',
                            values='Count',
                            title='Distribution of Variable Categories in Dataset'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # If no API key, provide basic data insights
                    st.info("Using basic analysis (Gemini API key not provided)")
                    
                    # Simple rule-based responses
                    if "highest" in query_lower and "income" in query_lower:
                        income_df = df[df['Variable_name'] == 'Total income'].sort_values('Value', ascending=False)
                        if not income_df.empty:
                            highest_industry = income_df.iloc[0]['Industry_name_NZSIOC']
                            highest_value = income_df.iloc[0]['Value']
                            
                            st.write(f"The industry with the highest total income is **{highest_industry}** with ${highest_value} million.")
                            
                            fig = px.bar(
                                income_df.head(10), 
                                x="Industry_name_NZSIOC", 
                                y="Value", 
                                title="Total Income by Industry (Top 10)",
                                labels={"Industry_name_NZSIOC": "Industry", "Value": "Income (millions)"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No income data found in the dataset.")
                    else:
                        st.write("I can answer basic questions about the data. Try asking about the highest income or other simple queries.")
                        st.write("For more advanced analysis, please add a Gemini API key to your secrets.toml file.")
                        
                        # Show dataset summary
                        st.subheader("Dataset Summary")
                        category_counts = df['Variable_category'].value_counts().reset_index()
                        category_counts.columns = ['Category', 'Count']
                        
                        fig = px.bar(
                            category_counts,
                            x='Category',
                            y='Count',
                            title='Variable Categories in Dataset'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Please try a different question or check your API key configuration.")

if __name__ == "__main__":
    main()