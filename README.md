# FraudShield: Enhanced Credit Card Fraud Detection Using Machine Learning

## Introduction

Credit card fraud poses a significant threat to financial institutions and consumers worldwide, leading to substantial financial losses annually. As digital transactions become increasingly prevalent, the need for robust fraud detection systems is more critical than ever. **FraudShield** is a machine learning project aimed at detecting fraudulent credit card transactions with high accuracy, thereby enhancing security and trust in financial systems.

**FraudShield** addresses this need by leveraging machine learning algorithms to detect fraudulent credit card transactions. By analyzing transaction patterns, the project aims to differentiate between legitimate and fraudulent activities effectively.

## Project Overview

The objective of this project is to develop and evaluate machine learning models that can effectively distinguish between legitimate and fraudulent credit card transactions. By leveraging historical transaction data and applying various machine learning algorithms, FraudShield seeks to identify patterns indicative of fraud, enabling proactive prevention of unauthorized activities.

**FraudShield** addresses this need by leveraging machine learning algorithms to detect fraudulent credit card transactions. By analyzing transaction patterns, the project aims to differentiate between legitimate and fraudulent activities effectively.

## Project Goals

The primary objective of this project is to develop and evaluate machine learning models capable of accurately detecting fraudulent credit card transactions. Specific goals include:

- Implement multiple machine learning algorithms for fraud detection.
- Compare the performance of these models to identify the most effective approach.
- Provide statistical analyses and visualizations to support findings.
- Explore previous literature and methodologies in the field of fraud detection.

## Data Source

The dataset utilized in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and contains credit card transactions made by European cardholders over two days in September 2013. The dataset comprises 284,807 transactions and includes the following features:

- **Time**: The seconds elapsed between each transaction and the first transaction.
- **V1** to **V28**: Principal components obtained from Principal Component Analysis (PCA) to protect sensitive information.
- **Amount**: The transaction amount.
- **Class**: The target variable, where 0 represents legitimate transactions and 1 represents fraudulent transactions.

**Note**: Due to confidentiality issues, the original feature names and background information are not provided.

## Algorithms Implemented

The following machine learning algorithms were implemented and compared:

1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Machine (SVM)**
3. **Logistic Regression**
4. **Decision Tree**

## Methodology

1. **Data Preprocessing**:
   - **Data Cleaning**: Ensured there were no missing or null values in the dataset.
   - **Feature Scaling**: Applied scaling to the `Amount` and `Time` features to standardize the data.
   - **Handling Imbalanced Data**: Addressed the class imbalance using techniques such as under-sampling and over-sampling.

2. **Model Training and Evaluation**:
   - Split the dataset into training and testing sets.
   - Trained each model using the training set.
   - Evaluated model performance using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

3. **Comparison of Models**:
   - Analyzed the performance of each model to determine the most effective algorithm for fraud detection.

## Results

### Model Performance Comparison

| Algorithm             | Accuracy (%) |
|-----------------------|--------------|
| K-Nearest Neighbors   | **~100.00**   |
| Decision Tree         | **~100.00**   |
| Support Vector Machine| 97.59        |
| Logistic Regression   | 93.51        |

- **K-Nearest Neighbors (K=3 and K=7)**: Both models achieved 100% accuracy on the training set, correctly identifying the majority of fraudulent transactions with minimal misclassifications.
- **Decision Tree**: Also achieved 100% accuracy, indicating its effectiveness in capturing decision rules for fraud detection.
- **Support Vector Machine**: Achieved 97.59% accuracy, demonstrating strong performance but slightly lower than KNN and Decision Tree.
- **Logistic Regression**: Achieved 93.51% accuracy, suggesting that while effective, it may not capture non-linear patterns as well as other algorithms.

## Future Work

To enhance the FraudShield project further, the following steps are proposed:

- **Dataset Expansion**: Apply the models to larger and more diverse datasets to improve generalization.
- **Algorithm Optimization**: Explore advanced algorithms like Random Forests, Gradient Boosting Machines, or Deep Learning techniques.
- **Real-Time Detection**: Develop real-time processing capabilities to detect fraud as transactions occur.
- **Feature Engineering**: Incorporate additional features such as transaction location data to improve model accuracy.
- **Model Deployment**: Implement the model into a production environment using tools like Flask or Django for practical application.

## Conclusion

FraudShield successfully demonstrates the application of machine learning algorithms in detecting credit card fraud. By comparing multiple models, the project identifies the most effective techniques for distinguishing fraudulent transactions, contributing to improved security measures in financial institutions and enhancing customer trust.

## Repository Structure

- **README.md**: Project overview and documentation.
- **Datasets**: Contains the dataset used for training and testing the models.
- **Notebooks**:
  - `K-Nearest Neighbor.ipynb`
  - `Support Vector Machines.ipynb`
  - `Logistic Regression.ipynb`
  - `Decision Tree.ipynb`
- **Results**: Contains evaluation reports and confusion matrices.

## Contributing

Contributions to FraudShield are welcome. If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Contact

Developed by [shxu7788](https://github.com/shxu7788). For any inquiries or support, please contact [shxu7788](https://github.com/shxu7788).
