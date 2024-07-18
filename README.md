# Movie Review Sentiment Analysis App

This application is designed to analyze the sentiment of movie reviews using a pre-trained model. It utilizes TensorFlow and Streamlit to provide a simple and interactive web interface where users can input a movie review and receive an analysis indicating whether the sentiment of the review is positive or negative, along with a confidence score.

## Features

- **Sentiment Analysis**: Determine whether a movie review is positive or negative.
- **Confidence Score**: Along with the sentiment, the app provides a confidence score to indicate the certainty of the analysis.
- **Interactive Web Interface**: Built with Streamlit, the app offers an easy-to-use interface for users to input their reviews and receive instant feedback.

## Installation

To run this application, you will need Python installed on your system. The app depends on several Python libraries, including TensorFlow, NumPy, and Streamlit. You can install all the required dependencies by running:

```bash
pip install tensorflow numpy streamlit
```

## Usage

1. Clone this repository to your local machine.
2. Navigate to the directory containing the app (`app.py`).
3. Run the app using Streamlit:

```bash
streamlit run app.py
```

4. Open your web browser and go to the address shown in your terminal (usually `http://localhost:8501`).
5. Enter a movie review in the text area provided and click the "Classify" button to analyze the sentiment.

## How It Works

The application uses a pre-trained TensorFlow model to analyze the sentiment of movie reviews. It preprocesses the input text to match the model's expected format, including tokenizing the text and padding it to a fixed length. The model then predicts the sentiment of the review, classifying it as either positive or negative.

## Contributing

Contributions to improve the app are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is open-source and available under the MIT License.