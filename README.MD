# AnalysisX

## Overview

AnalysisX is an application that fetches relevant data from social media APIs and uses a deep learning model to predict the sentiment of specific statements. The main goal of the project is to provide sentiment analysis for real-time data, sorted based on relevant hashtags. The application uses a Recurrent Neural Network (RNN) model trained on historical data to predict sentiment scores between 0 and 1, where 0 represents a negative sentiment and 1 represents a positive sentiment.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/AISH-1/AnalysisX.git`
2. Navigate to the project directory: `cd AnalysisX`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

To use the AnalysisX application, follow these steps:

1. Run the Flask application using the following command:

```bash
python app.py
```

2. Access the application by navigating to `http://127.0.0.1:5000/` in your web browser.

3. Log in using your login ID and password to access the secured website.

4. Enter a statement or text in the provided text area.

5. Submit the form to get the sentiment analysis prediction for the entered statement.

## Data

The application uses data fetched from social media APIs to train the deep learning model. The historical data is preprocessed to clean unwanted characters and then used to train an RNN model to predict sentiment scores.

## Dependencies

The following external dependencies and libraries are required to run the application:

- Flask
- pandas
- numpy
- tensorflow
- re

These dependencies are listed in the `requirements.txt` file and can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, execute the following command in your terminal or command prompt:

```bash
python app.py
```

The Flask development server will start, and you can access the application by navigating to `http://127.0.0.1:5000/` in your web browser.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have any suggestions, improvements, or bug fixes. We appreciate your help in making AnalysisX better for everyone.

When contributing, please follow these guidelines:

- Before starting any significant changes, create an issue to discuss the proposed improvement or bug fix.
- Fork the repository and create a new branch for your changes.
- Ensure that your code adheres to the project's coding standards and style guidelines.
- Write clear and concise commit messages to describe your changes.
- Test your changes thoroughly to avoid introducing new issues.
- Once you're ready, please submit a pull request, and we'll go over your changes as soon as possible.

We value every contribution and strive to maintain a friendly and inclusive community. Thank you for your support and efforts in making AnalysisX a better project!

## License

This project is licensed under the [MIT License](LICENSE).
