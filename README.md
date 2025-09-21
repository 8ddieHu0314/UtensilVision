# Utensil Vision: Cornell Dining Application

Utensil Vision utilizes fine-tuned YOLOv11 model to detect forks, knives, and spoons, streamlining the dish return process in Cornell dining hall. It uses computer vision and real-time data logging to detect, classify, and count returned utensils—helping reduce queue times, eliminate manual sorting, and predict future utensil demand to avoid shortages and surpluses in the utensils prepared.

## Features

- **Image Recognition**: Detects and classifies utensils (forks, spoons, knives) in real-time as they are returned.
- **Real-Time Utensil Counter**: Tracks how many of each utensil type are returned throughout the day.
- **Hardware Integration (Future)**: Supports integration with servo motors and sorting mechanisms for automatic physical separation of utensils.

## Installation

1. Clone the repository:

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the sorting application:

    ```bash
    streamlit run app.py
    ```

3. The app will:
   - Detect and classify each utensil.
   - Count the type of utensil returned.
   - Display current utensil statistics and trends.

## Model

The dataset used and the model versions can be found [here](https://app.roboflow.com/genai-gkvsb/utensils-jabsv/models)