# Data Science Experiment 3

This repository contains the code and dataset for Experiment 3 of the Data Science Fundamentals with Python course.

## Description

In this experiment, we:
- Loaded the "Heart Disease" dataset from the UCI Machine Learning Repository.
- Performed data preprocessing by encoding categorical variables.
- Converted categorical variables to dummy variables.
- Applied label encoding to binary categorical variables.

## Files

- `experiment_3.ipynb`: The Jupyter Notebook containing the experiment code.
- `encoded_heart_disease_data.csv`: The encoded dataset generated from the experiment.

## Usage

To run the code, open the `experiment_3.ipynb` file in Google Colab or Jupyter Notebook.

## Steps to Reproduce

1. **Set up Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Create a new notebook.

2. **Import Necessary Libraries**:
   - Start by importing the required libraries.

    ```python
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    ```

3. **Load the Dataset**:
   - Use the URL of the dataset from the UCI ML repository. For this example, let's use the "Heart Disease" dataset.

    ```python
    # Load the dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
               'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv(url, header=None, names=columns, na_values='?')
    ```

4. **Display the Dataset**:
   - Show the first few rows of the dataset to verify it loaded correctly.

    ```python
    data.head()
    ```

5. **Check for Missing Values**:
   - Identify and count missing values in the dataset.

    ```python
    data.isnull().sum()
    ```

6. **Drop Rows with Missing Values**:
   - Remove rows with missing values.

    ```python
    data_cleaned = data.dropna()
    data_cleaned.isnull().sum()
    ```

7. **Convert Categorical Variables to Dummy Variables**:
   - Use `pd.get_dummies` to convert categorical variables into dummy variables.

    ```python
    data_encoded = pd.get_dummies(data_cleaned, columns=['cp', 'restecg', 'slope', 'thal'])
    data_encoded.head()
    ```

8. **Label Encode Binary Variables**:
   - Use `LabelEncoder` to encode binary categorical variables.

    ```python
    labelencoder = LabelEncoder()
    data_encoded['sex'] = labelencoder.fit_transform(data_encoded['sex'])
    data_encoded['fbs'] = labelencoder.fit_transform(data_encoded['fbs'])
    data_encoded['exang'] = labelencoder.fit_transform(data_encoded['exang'])
    data_encoded.head()
    ```

9. **Save and Export the Encoded Dataset**:
    - Save the encoded dataset to a CSV file and download it locally.

    ```python
    data_encoded.to_csv('encoded_heart_disease_data.csv', index=False)
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
