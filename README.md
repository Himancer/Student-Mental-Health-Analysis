# Student-Mental-Health-Analysis


## Overview

This code performs the following tasks:
- Loads a dataset from a CSV file.
- Preprocesses the data by handling missing values and converting categorical variables to numerical values.
- Explores the data through various visualizations and statistical analyses.
- Builds machine learning models to predict the total number of mental health issues each student has.
- Evaluates model performance using different metrics.

This analysis is valuable for understanding factors related to student mental health and developing predictive models for mental health issues.

## Usage

To run the code, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Himancer/Student-Mental-Health-Analysis.git

cd Student-Mental-Health-Analysis

pip install -r requirements.txt

python Student-Mental-Health-Analysis.py


5. Include information about the dataset used, its source, and a brief description of the columns in the dataset.

```markdown
## Dataset

The dataset used in this analysis is named "Student Mental Health" and is provided in the `Student Mental health.csv` file. This dataset contains information related to student mental health, including factors such as age, course, depression, anxiety, panic attacks, and more. The data was collected through a survey or questionnaire.

### Columns

- `Choose your gender`: Gender of the student.
- `What is your course?`: The course or program the student is enrolled in.
- `Your current year of Study`: The current year of study.
- `Marital status`: Marital status of the student.
- `Do you have Depression?`: Binary response to whether the student has depression.
- `Do you have Anxiety?`: Binary response to whether the student has anxiety.
- `Do you have Panic attack?`: Binary response to whether the student has panic attacks.
- `Did you seek any specialist for a treatment?`: Whether the student sought specialist treatment.
- `Age`: Age of the student.
- `Total Mental Health Issues`: A new feature indicating the total number of mental health issues each student has.
- `CGPA Midpoint`: Midpoint value for CGPA, derived from the original CGPA range.
- `Study Year`: Numerical representation of the current year of study.

The dataset can be found in the `Student Mental health.csv` file.

## Data Analysis

The code includes various data analysis and visualization steps:

- Data exploration using summary statistics.
- Distribution plots for numerical variables.
- Correlation matrix and heatmap to visualize relationships between variables.
- Box plots to understand the relationships between categorical and numerical variables.
- Count plots for categorical columns.

These analyses provide insights into the dataset and help in understanding the factors related to student mental health.

## Machine Learning Models

The code uses the following machine learning models to predict the total number of mental health issues each student has:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The models are trained on the dataset and evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

These metrics help assess the performance of the models in predicting mental health issues.

## Additional Information

- This code is part of a project related to student mental health analysis.
- Feel free to contribute to this project by opening issues or pull requests on GitHub.

## Acknowledgments

The data used in this analysis may have been collected from various sources or surveys. Please refer to the dataset source for more details.

## Contact

For any questions or inquiries, you can contact Hmanshu Pandey at hpin.2206@gmail.com.

