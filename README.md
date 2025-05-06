# 📰 Fake News Detection

A machine learning project designed to classify news articles as **Fake** or **Real** using natural language processing techniques and supervised learning algorithms.

## 🚀 Live Demo

Experience the application live:

👉 [Streamlit App on Hugging Face](https://huggingface.co/spaces/Saber-0722/Fake-News-Detection)

## 📂 Project Structure

```
├── data/
│   ├── true.csv
│   ├── fake.csv
│   ├── cleaned_data.csv
│   ├── vaders.csv
│
├── notebooks/
│   └── EDA.ipynb
│   └── data_preprocessing.ipynb
│   └── train_model.ipynb
├── Outputss/
│   └── plots/ 
│   └── models/
├── src/
│   ├── app.py
├── front.py
├── requirements.txt
└── README.md
```

* **data/**: Contains the raw data and cleaned data.
* **notebooks/**: Jupyter notebooks for exploratory data analysis and model development and feature engineering.
* **src/**: Python scripts for the streamlit app.
* **Outputs/**: Contains plots and models saved into the respective sub folders.
* **requirements.txt**: Lists all Python dependencies.

## 📊 Features

* **Data Preprocessing**: Cleans and prepares text data for modeling.
* **Feature Extraction**: Utilizes TF-IDF vectorization to convert text into numerical features.
* **Model Training**: Implements classifiers like Logistic Regression.
* **Model Evaluation**: Assesses models using metrics such as F1-score and confusion matrix.
* **Feature Importance**: Identifies top 50 influential words in each class.
* **Visualization**: Generates confusion matrix.
* **Web Interface**: Provides a user-friendly Streamlit app for real-time predictions.

## 🛠️ Setup Instructions

### Prerequisites

* Python 3.11 or higher
* pip package manager

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Saber0722/Fake_News_Detection.git
   cd Fake_News_Detection
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**

   Navigate to the src folder before running the app

   ```bash
   cd src
   ```
   ```bash
   streamlit run app.py
   ```

## 🧪 Usage

Once the Streamlit app is running:

1. Navigate to the local URL provided by Streamlit (usually `http://localhost:8501/`).
2. Enter a news article or statement into the input field.
3. Click the "Classify" button to determine if the news is **Fake** or **Real**.
4. View the prediction result.

## 📁 Dataset

The project utilizes the following dataset from [kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection).

* **Description**: Contains around 23000 entires for fake data and 21000 entries for real data.


## 📈 Model Performance

**Logistic Regression** model is trained and achieved accuracy score of 0.99. The model demonstrates robust performance in distinguishing between fake and real news articles.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
