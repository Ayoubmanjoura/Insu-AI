# Insulin Prediction AI

This repository contains a machine learning model designed to predict the optimal amount of insulin to be injected based on various inputs, such as blood glucose levels, it movement (if it is stable, rising or going down), physical effort, if any meal has been taken and other.

## Table of Content
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Ayoubmanjoura/Insu-AI.git
    ```

2. Install the required dependencies:
    ```bash
    cd Insu-AI
    pip install -r requirements.txt
    ```

3. (Optional) Set up a virtual environment to isolate the project dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

## Usage

To start predicting insulin dosage:

1. Ensure you have the necessary input data (e.g., blood glucose levels, carbohydrate intake, etc.).

2. Run the prediction script:
    ```bash
    python app_deployment.py
    ```
3. Give the AI the input as asked.
4. The model will return the recommended insulin dosage based on the inputs provided.
