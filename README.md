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
    git clone https://github.com/tdeepak509/duplicate_CQA_detection.git
    ```
2. Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```

### Data
For this research, two separate datasets will be utilised. The first dataset will be fetched from Quora (general domain CQA website), and the second dataset will be retrieved from Stack Overflow (a technical domain CQA website). Both datasets have at least three columns, the first two of which are for the original and duplicate questions, and the third column is the target label, which indicates whether or not the question pair is duplicate.

1. Dataset 1 – Quora
The dataset for the CQA website Quora is fetched from the Initial Quora Dataset release which is hosted on [Amazon S3](http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv). The dataset is present as a .tsv file consisting of 404,290 no of rows, 6 data fields, and the overall size of this file is 54.8 MB. Each line in the dataset consists of potential duplicate question pair. 

2. Dataset 2 – StackOverflow
The dataset of Stack Overflow is fetched by executing a Big Data Query against the Stack Exchange Data Explorer (SEDE) (Data Stack Exchange, n.d.) which in turn fetches data from Stack Exchange’s data dump repository. This data repository is provided by Creative Commons Data Dump Service. The SEDE is Stack Exchange's official file browser, from where the users can run queries to access the data. The questions, answers, changes, reviews, users, badges, tags, and other information that you see on Stack Exchange are all kept in a relational database. And in the Stack Exchange’s data repository, stack overflow alone has 23M questions, which is a lot of information. 

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
