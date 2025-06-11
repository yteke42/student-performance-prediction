# How to Use the Student Performance Prediction App

## Overview
This application helps predict student performance and identify at-risk students using machine learning. It can process individual student data, bulk student grades, and raw Moodle logs to provide predictions and recommendations.

## Getting Started

### 1. Selecting a Model
- Navigate to the "Model Selection" tab
- Choose from available trained models:
  - Random Forest
  - XGBoost
  - Neural Network
- The app will automatically load the selected model

### 2. Individual Student Prediction

#### Using the Input Form:
1. Go to the "Predict Performance" tab
2. Fill in the required student information:
   - Student ID
   - Course ID
   - Current Grade
   - Number of Assignments Completed
   - Number of Quizzes Completed
   - Number of Forum Posts
   - Number of Resources Accessed
   - Number of Days Active
3. Click "Predict Performance"
4. View the results:
   - Predicted Grade
   - Risk Level (Low/Medium/High)
   - Performance Insights
   - Recommended Actions

#### Using Excel Upload:
1. Go to the "Predict Performance" tab
2. Click "Upload Excel File" in the "Bulk Prediction" section
3. Your Excel file should have these columns:
   - student_id
   - course_id
   - current_grade
   - assignments_completed
   - quizzes_completed
   - forum_posts
   - resources_accessed
   - days_active
4. Click "Predict Performance"
5. Download results as Excel file

### 3. Processing Raw Moodle Logs

1. Go to the "Raw Logs Processing" tab
2. Upload your Moodle log file (CSV format)
3. The app will:
   - Preprocess the logs
   - Extract relevant features
   - Generate predictions
4. View the results:
   - Preprocessing status
   - Feature extraction summary
   - Prediction results
5. Download the processed data and predictions

### 4. Understanding the Outputs

#### Performance Prediction
- Predicted Grade: The model's estimate of the student's final grade
- Risk Level:
  - Low: Student is likely to perform well
  - Medium: Student may need some support
  - High: Student is at risk of poor performance

#### Insights and Recommendations
- Performance Insights: Analysis of the student's current performance
- Recommended Actions: Specific steps to improve performance
- Historical Data: Comparison with similar students

### 5. Downloading Results

#### For Individual Predictions:
- Results are displayed directly in the app
- Click "Download Results" to save as CSV

#### For Bulk Predictions:
- Results are automatically compiled into an Excel file
- Click "Download Results" to save the file
- The downloaded file includes:
  - Original data
  - Predictions
  - Risk levels
  - Recommendations

## Input Requirements

### Excel File Format
Required columns for bulk prediction:
- student_id: Unique identifier for each student
- course_id: Course identifier
- current_grade: Current grade (0-100)
- assignments_completed: Number of completed assignments
- quizzes_completed: Number of completed quizzes
- forum_posts: Number of forum posts
- resources_accessed: Number of accessed resources
- days_active: Number of active days

### Raw Logs Format
The Moodle logs should be in CSV format with these columns:
- user_id
- course_id
- timestamp
- action
- resource_type
- grade (if applicable)

## Tips for Best Results
1. Ensure all required fields are filled
2. Use consistent formatting for student and course IDs
3. Verify that grades are within the 0-100 range
4. For bulk predictions, check that your Excel file matches the required format
5. For raw logs, ensure the CSV file is properly formatted and complete

## Troubleshooting
- If predictions seem inaccurate, try:
  - Verifying input data accuracy
  - Using a different model
  - Checking for missing or incorrect values
- For upload issues:
  - Verify file format (CSV/Excel)
  - Check file size (should be under 100MB)
  - Ensure all required columns are present 