# Wine Classification with Naive Bayes

A Python machine-learning project that classifies wines into three classes using a Gaussian Naive Bayes model.

## Dataset

The Wine Recognition dataset contains chemical-analysis measurements of wines grown in the same region of Italy.

- Samples: 178
- Input features: 13 continuous chemical measurements
- Output classes: 3 wine cultivars (Class 1, Class 2, and Class 3)

Examples of features are Alcohol, Malic Acid, Color Intensity, Hue, and Proline.

## Workflow

1. Load the `wine.data` file using Pandas.
2. Check the dataset shape, missing values, class counts, and summary statistics.
3. Split the data into training and testing sets using stratified sampling.
4. Train a `GaussianNB` model from scikit-learn.
5. Predict wine classes for the test data.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## How It Works

Bayes' theorem is:

\[
P(C \mid X) = \frac{P(X \mid C) \times P(C)}{P(X)}
\]

Naive Bayes calculates the probability of each class and assigns a sample to the class with the highest probability. Gaussian Naive Bayes is suitable here because the input features are continuous numerical values.

## Visualizations

- Wine class-distribution bar chart
- Alcohol versus Color Intensity scatter plot
- Feature-distribution histograms
- Correlation heatmap
- Confusion matrix
- Precision, Recall, and F1-score chart
- PCA visualization of wine classes

## Notes

Naive Bayes is fast and useful as a baseline classifier. It works best when features are approximately independent. It may struggle when input features are strongly correlated, because its independence assumption is then not fully satisfied.

## Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Upload `wine.data` when prompted.
3. Run all cells from top to bottom.
