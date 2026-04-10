# TruthReview

**TruthReview** is an NLP-based intelligent application that analyzes hotel reviews to predict whether a review is **fake (deceptive)** or **real (truthful)**. This project is developed as the final assignment for the **Introduction to Large Language Models** course.

## Project Objective

Online reviews are important sources of information that directly influence users' purchasing and preference decisions. However, some reviews on digital platforms may not reflect genuine user experiences and may be written for manipulative purposes. The goal of this project is to develop a system that can automatically distinguish between fake and real reviews by analyzing review texts.

TruthReview classifies hotel reviews using Natural Language Processing (NLP) methods and provides the user with a prediction of whether a review is **Fake** or **Real**.

## Problem Definition

Fake reviews:

- can cause users to make incorrect decisions,
- can create misleading perceptions about businesses,
- can reduce the credibility of online platforms,
- can overshadow genuine user experiences.

This project addresses the fake review detection problem as a **binary text classification** problem.

## Project Scope

This application:

- takes a hotel review from the user,
- applies preprocessing steps on the text,
- analyzes the review with the selected NLP model,
- generates a prediction of whether the review is **Fake** or **Real**.

Although this project focuses on the fake review detection problem in general, it is initially limited to the **hotel reviews domain** due to the available dataset.

## Dataset

The **Deceptive Opinion Spam Corpus** dataset is used in this project. This dataset consists of hotel reviews labeled as **truthful** and **deceptive**.

Reasons for choosing this dataset:

- it is directly suitable for the fake review detection problem,
- it is a well-known and widely used dataset in academic research,
- it is of manageable size within the project timeline,
- it offers a clean and understandable structure for classification experiments.

## Course Alignment

This project is aligned with the requirements expected within the course scope. It particularly covers the following areas:

- natural language processing fundamentals,
- text preprocessing,
- text representation,
- classification model development,
- model training,
- performance evaluation,
- developing a working intelligent application.

In this regard, the project is evaluated under **Track 1: Natural Language Processing (NLP)**.

## Planned Approach

The following process is generally followed in the project:

1. Examining and preparing the dataset  
2. Preprocessing the review texts  
3. Representing texts numerically  
4. Training a baseline classification model  
5. Evaluating model performance  
6. Integrating the trained model into a simple application interface  

In the initial phase, a classical baseline approach is preferred in order to build an explainable and manageable starting system.

## Model Approach

The following baseline model is used in the initial phase:

- **TF-IDF + Logistic Regression**

The following alternatives may also be evaluated if necessary:

- TF-IDF + Naive Bayes
- BERT-based text classification

This approach allows both basic and more advanced NLP methods to be compared.

## Evaluation Metrics

The following metrics are used to evaluate model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics allow not only the overall accuracy of the model but also its ability to distinguish fake reviews to be analyzed in more detail.

## Expected Outputs

The following outputs are targeted at the end of the project:

- a working NLP-based application,
- a fake / real review classification system,
- experiment outputs including performance results,
- a comprehensive project report,
- code repository link,
- a short video demo presentation.

## Project Folder Structure

```text
TruthReview/
│
├── README.md
│
├── src/
├── data/
├── notebooks/
├── results/
└── report/
```
