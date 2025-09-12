import streamlit as st
import imageio
import time
import cv2  # Only for frame conversion if needed
import os

# Function to initialize the webcam reader
def start_webcam():
    try:
        reader = imageio.get_reader('<video0>', 'ffmpeg')  # '<video0>' for default webcam (adjust for your OS if needed)
        return reader
    except Exception as e:
        st.error(f"Error starting webcam: {e}")
        return None

# Main app
st.title("Real-Time Utensil Detection")

# Placeholder for the video feed
frame_placeholder = st.empty()

# Button to start/stop
if 'reader' not in st.session_state:
    st.session_state.reader = None
    st.session_state.running = False

if st.button("Start Webcam") and not st.session_state.running:
    st.session_state.reader = start_webcam()
    if st.session_state.reader:
        st.session_state.running = True
        st.rerun()  # Rerun to enter the loop

if st.button("Stop Webcam") and st.session_state.running:
    if st.session_state.reader:
        st.session_state.reader.close()
    del st.session_state.reader
    st.session_state.running = False
    frame_placeholder.empty()  # Clear the display
    st.rerun()

# Real-time loop (only runs if started)
if st.session_state.running and 'reader' in st.session_state:
    while st.session_state.running:
        try:
            # Grab a frame
            frame = st.session_state.reader.get_next_data()  # NumPy array (RGB)

            frame = cv2.flip(frame, 1)

            # Display the frame (resize for UI)
            frame_placeholder.image(frame, channels="RGB", width=200, use_container_width=False)

            #time.sleep(0.001)  # ~20 FPS to avoid CPU overload
        except Exception as e:
            st.warning(f"Error in loop: {e}")
            break  # Exit on error

    # Cleanup if loop exits
    if 'reader' in st.session_state:
        st.session_state.reader.close()
        del st.session_state.reader
    st.session_state.running = False
    frame_placeholder.empty()