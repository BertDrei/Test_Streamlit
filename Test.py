import streamlit as st 
import pandas as pd

# Load data
data = pd.read_csv("data.csv")

# Set a simple title
st.title('New Zealand Industry Financial Data')

# Display raw data in a collapsible section
with st.expander("View Raw Data"):
    st.write(data)

# Create a simple chart
st.subheader("Basic Financial Visualization")

# Get unique industries
industries = sorted(data['Industry_name_NZSIOC'].unique())
selected_industry = st.selectbox("Select an Industry:", industries)

# Get unique financial metrics
metrics = sorted(data['Variable_name'].unique())
selected_metric = st.selectbox("Select a Financial Metric:", metrics)

# Filter data based on selection
filtered_data = data[(data['Industry_name_NZSIOC'] == selected_industry) & 
                     (data['Variable_name'] == selected_metric)]

# Make sure to convert values to numeric for plotting
filtered_data['Value'] = pd.to_numeric(filtered_data['Value'], errors='coerce')

if not filtered_data.empty:
    # Create a bar chart using streamlit's native chart functionality
    st.bar_chart(filtered_data.set_index('Year')['Value'])
    
    # Display the selected data as a table
    st.write(f"{selected_metric} for {selected_industry}")
    st.write(filtered_data[['Year', 'Value', 'Units']])
else:
    st.warning("No data available for this selection.")