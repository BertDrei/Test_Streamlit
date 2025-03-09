import streamlit as st
import requests
import json

st.title("Gemini API Key Tester")

# Store API key securely
api_key = st.text_input("Enter your Gemini API key:", value="AIzaSyB7r8xNPUW0X4vhVKObaUqqBjTt-UrRzo4", type="password")
st.caption("Your API key is stored securely and not logged.")

# Text prompt input
prompt = st.text_area("Enter a prompt to test:", value="Explain how AI works", height=100)

if st.button("Test API Key"):
    if not api_key:
        st.error("Please enter an API key")
    else:
        st.info("Testing API key...")
        
        # API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        # Request headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Request data
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            # Make the API request
            with st.spinner("Waiting for API response..."):
                response = requests.post(url, headers=headers, data=json.dumps(data))
            
            # Display response status
            if response.status_code == 200:
                st.success(f"API request successful! (Status code: {response.status_code})")
                
                # Parse and display the response
                response_data = response.json()
                st.subheader("API Response:")
                
                # Extract the text from the response
                try:
                    text_response = response_data["candidates"][0]["content"]["parts"][0]["text"]
                    st.write(text_response)
                except KeyError:
                    st.json(response_data)  # Display raw JSON if structure differs
                    
            else:
                st.error(f"API request failed with status code: {response.status_code}")
                st.subheader("Error details:")
                st.json(response.json())
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.divider()
st.caption("Note: Run this app with `streamlit run Test.py` in your terminal")