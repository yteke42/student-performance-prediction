# Student Performance Prediction App

A Streamlit-based application that uses machine learning to predict student performance and identify at-risk students. The app processes student data, Moodle logs, and course activities to provide actionable insights and recommendations.

## Features

- **Multiple Model Support**
  - Random Forest
  - XGBoost
  - Neural Network
  - Easy model switching and comparison

- **Flexible Input Methods**
  - Individual student prediction
  - Bulk prediction via Excel upload
  - Raw Moodle logs processing

- **Comprehensive Analysis**
  - Performance predictions
  - Risk level assessment
  - Personalized recommendations
  - Historical data comparison

- **User-Friendly Interface**
  - Interactive input forms
  - Real-time predictions
  - Visual performance insights
  - Easy result downloads

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the App

1. Activate your virtual environment (if not already activated):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Start the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to:
```
http://localhost:8501
```

## Project Structure

```
student-performance-prediction/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── how_to_use.md         # User guide
├── models/               # Trained model files
│   ├── random_forest/
│   ├── xgboost/
│   └── neural_network/
├── data/                 # Data storage
│   ├── raw/             # Raw input data
│   └── processed/       # Processed data
└── logs/                # Application logs
```

## Usage

For detailed instructions on using the application, please refer to the [How to Use Guide](how_to_use.md).

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow
- Other dependencies listed in `requirements.txt`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Streamlit for the web framework
- Scikit-learn for machine learning tools
- XGBoost for gradient boosting implementation
- TensorFlow for neural network capabilities

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/student-performance-prediction](https://github.com/yourusername/student-performance-prediction) 