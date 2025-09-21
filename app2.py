import streamlit as st
import pandas as pd
import random

# Sample data generation function
def generate_sample_data(dining_halls):
    data = {}
    sessions = ['Breakfast', 'Lunch', 'Dinner']
    for hall in dining_halls:
        hall_data = {session: random.randint(50, 300) for session in sessions}
        data[hall] = hall_data
    return data

# List of Cornell dining halls (example list, you can expand this)
dining_halls = [
    'Bethe House',
    'Robert Purcell Community Center',
    'North Star Dining Room',
    'Okenshields',
    'Risley Dining',
    'Morrison Dining',
    'Jansen\'s at Bethe House',
    '104West!'
]

# Generate sample data
utensils_data = generate_sample_data(dining_halls)

# Streamlit app
st.title('Cornell Dining Halls Utensils Tracker')

st.write('This app displays the number of utensils collected in different dining halls at Cornell. Select a dining hall to view details for each session.')

# Create tabs for each dining hall
tab_titles = dining_halls
tabs = st.tabs(tab_titles)

for i, tab in enumerate(tabs):
    with tab:
        hall = dining_halls[i]
        st.subheader(f'{hall} - Utensils Collected')
        
        # Display data in a table
        df = pd.DataFrame(list(utensils_data[hall].items()), columns=['Session', 'Utensils Collected'])
        st.table(df)
        
        # Optional: Display a bar chart
        st.subheader('Visualization')
        st.bar_chart(df.set_index('Session'))

st.write('Note: This is sample data. In a real app, replace the generate_sample_data function with actual data sources (e.g., from a database or API).')

# Instructions to run the app
st.sidebar.title('How to Run This App')
st.sidebar.write('Save this code as app.py and run `streamlit run app.py` in your terminal. Access it via a web browser on your mobile device for a mobile-friendly experience.')