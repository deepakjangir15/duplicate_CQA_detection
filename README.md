# Project Title
Duplicate Question Detection for CQA Websites

## Project Description
The project aims to detect duplicate questions from CQA (Community Question Answering) websites such as Quora and Stack Overflow using deep learning models. The project will evaluate the performance of various deep learning models such as Convolutional Neural Networks (CNNs) and Transformer models like BERT and RoBERTa. The impact of hyperparameters such as number of layers, learning rate, and batch size on the performance of existing and new deep learning models will be explored. Additionally, the project will compare the performance of deep learning models to traditional machine learning models like random forest and SVM. The use of transfer learning with pre-trained language models will also be explored to improve model accuracy. Advanced feature engineering techniques such as semantic analysis, topic modeling, and Named Entity Recognition (NER) will also be explored to improve model accuracy.

## Objectives
- Evaluate the performance of additional deep learning models, such as Convolutional Neural Networks (CNNs) and Transformer models like BERT and RoBERTa, for detecting duplicate question pairs on CQA websites.
- Explore the impact of hyperparameters for tuning on the performance of existing (LSTM) and new deep learning models, including number of layers, the learning rate, and the batch size.
- Compare the performance of the deep learning models to traditional machine learning models, such as random forest and SVM, to determine the most effective approach for detecting duplicate question pairs on CQA websites.
- Use transfer learning to leverage pre-trained language models and improve the accuracy of the deep learning models in detecting duplicate question pairs. For example, fine-tuning BERT for question pair similarity and transfer learning with RoBERTa.
- Utilize a large dataset for training and testing the models to improve their accuracy and generalization.
- Explore advanced feature engineering techniques, such as semantic analysis, topic modeling, and Named Entity Recognition (NER), to improve the accuracy of the models in detecting duplicate question pairs on CQA websites.

## Getting Started
### Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Spacy
- Gensim

### Installation
1. Clone the repo
    ```bash
    git clone https://github.com/<your_username>/duplicate-question-detection.git
    ```
2. Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```

### Data
The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/c/quora-question-pairs/data). The dataset contains question pairs labeled as duplicates or not.

## Usage
The project is divided into different Jupyter notebooks for each deep learning model and feature engineering technique. The notebooks can be run in the following order:

1. Data cleaning and preprocessing
2. Feature engineering
3. Traditional machine learning models
4. Deep learning models
5. Transfer learning

## Contributions
Contributions to the project are welcome. If you find any bugs or want to add new features, feel free to open an issue or create a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
