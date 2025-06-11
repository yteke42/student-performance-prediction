# Student Performance Prediction System

This is a Streamlit-based web application that predicts student performance using machine learning models. The system can analyze various factors affecting student performance and provide predictions based on different models.

## Features

- Multiple prediction models:
  - Full Model (includes midterm grades)
  - Early Warning Model (without midterm grades)
  - Custom Model training capability
- Real-time performance prediction
- Feature importance visualization
- Model training and evaluation tools
- Support for custom model training with user data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student_performance_prediction.git
cd student_performance_prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the application:
   - Select a prediction model
   - Input student data
   - View predictions and feature importance
   - Train custom models if needed

## Project Structure

- `app.py`: Main application file
- `models/`: Directory containing trained models
- `training_logs/`: Directory for storing training logs
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Data Requirements

The application can work with various types of student data, including:
- Attendance records
- Assignment submissions
- Quiz attempts
- Course module views
- Other activity logs
- Grade information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Uses scikit-learn for machine learning models
- Matplotlib for visualizations

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/student_performance_prediction](https://github.com/yourusername/student_performance_prediction) 