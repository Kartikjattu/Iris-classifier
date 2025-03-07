# Iris Classifier 🌸

## Description
This project implements a machine learning model to classify the Iris dataset using a **Random Forest Classifier**. The model is trained to predict the species of Iris flowers based on their **sepal and petal dimensions**.

## Features
- Loads and processes the **Iris dataset**
- Performs **exploratory data analysis** with visualizations
- Splits data into training and testing sets
- Trains a **Random Forest Classifier** for classification
- Evaluates model performance using **confusion matrix** and **classification report**

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Kartikjattu/iris-classifier.git
   cd iris-classifier
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the classifier:
   ```bash
   python iris_classifier.py
   ```

## Dataset
The **Iris dataset** consists of 150 samples with four features:
- Sepal length
- Sepal width
- Petal length
- Petal width
Each sample belongs to one of three species:
- **Setosa**
- **Versicolor**
- **Virginica**

## Model Training & Evaluation
- **Train-Test Split:** 70% training, 30% testing
- **Model Used:** Random Forest Classifier with 100 estimators
- **Performance Metrics:**
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

## Example Output
```
Confusion Matrix:
[[15  0  0]
 [ 0 14  1]
 [ 0  1 14]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        15
           1       0.93      0.93      0.93        15
           2       0.93      0.93      0.93        15

   accuracy                           0.96        45
  macro avg       0.96      0.96      0.96        45
 weighted avg       0.96      0.96      0.96        45
```

## Visualization
- **Pairplot of features** categorized by species
- **Heatmap of confusion matrix** to visualize classification results

## Contributing
Feel free to submit pull requests or open issues to enhance the model or visualizations.

## License
This project is licensed under the MIT License.

