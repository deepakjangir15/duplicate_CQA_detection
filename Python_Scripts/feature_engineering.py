import pandas as pd
import os
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import seaborn as sns
import helper as helper
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
import numpy as np

folder_path = 'visualizations'

train_df_balanced = pd.read_pickle('dataset/train_df_balanced.pkl')
token_feature_engineering = train_df_balanced.apply(
    helper.fetch_token_features, axis=1)


train_df_balanced["common_words_min"] = list(
    map(lambda x: x[0], token_feature_engineering))
train_df_balanced["common_words_max"] = list(
    map(lambda x: x[1], token_feature_engineering))
train_df_balanced["common_stop_min"] = list(
    map(lambda x: x[2], token_feature_engineering))
train_df_balanced["common_stop_max"] = list(
    map(lambda x: x[3], token_feature_engineering))
train_df_balanced["common_token_min"] = list(
    map(lambda x: x[4], token_feature_engineering))
train_df_balanced["common_token_max"] = list(
    map(lambda x: x[5], token_feature_engineering))
train_df_balanced["last_word_equal"] = list(
    map(lambda x: x[6], token_feature_engineering))
train_df_balanced["first_word_equal"] = list(
    map(lambda x: x[7], token_feature_engineering))


length_features = train_df_balanced.apply(helper.get_length_features, axis=1)

train_df_balanced['abs_length'] = list(map(lambda x: x[0], length_features))
train_df_balanced['mean_length'] = list(map(lambda x: x[1], length_features))
train_df_balanced['longest_substring_ratio'] = list(
    map(lambda x: x[2], length_features))

fuzzy_features = train_df_balanced.apply(helper.fetch_fuzzy_features, axis=1)

# Creating new feature columns for fuzzy features
train_df_balanced['fuzzy_ratio'] = list(map(lambda x: x[0], fuzzy_features))
train_df_balanced['fuzzy_partial_ratio'] = list(
    map(lambda x: x[1], fuzzy_features))
train_df_balanced['token_sort_ratio'] = list(
    map(lambda x: x[2], fuzzy_features))
train_df_balanced['token_set_ratio'] = list(
    map(lambda x: x[3], fuzzy_features))


# Create a correlation matrix to show the correlation between features and target variable
corr_matrix = train_df_balanced[['common_words_min', 'common_words_max',
                                 'common_stop_min', 'common_stop_max', 'is_duplicate']].corr()

# Create a heatmap to visualize the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# Save the image in the folder
image_path = os.path.join(
    folder_path, 'correlation between features and target variable.png')
plt.savefig(image_path)
# Clear the current figure
plt.clf()


# Create a pairplot to visualize the relationship between the features and target variable
sns.pairplot(train_df_balanced[['common_words_min', 'common_words_max', 'common_stop_min', 'common_stop_max', 'is_duplicate']],
             hue='is_duplicate')

# Set descriptive labels for axes and legend
plt.xlabel('Common Words Min')
plt.ylabel('Common Words Max')
plt.legend(title='Is Duplicate?')

# Add a title to the plot
plt.suptitle('Relationship between Features and Duplicate Detection')
# Save the image in the folder
image_path = os.path.join(
    folder_path, 'pairplot - features and target variable.png')
plt.savefig(image_path)
# Clear the current figure
plt.clf()

sns.pairplot(train_df_balanced[[
             'last_word_equal', 'first_word_equal', 'is_duplicate']], hue='is_duplicate')
image_path = os.path.join(
    folder_path, 'pairplot - features and target variable 2.png')
plt.savefig(image_path)
# Clear the current figure
plt.clf()

print('-------------------------------------------\n')

# Combine all the questions into a single list
all_questions = list(
    train_df_balanced['question1']) + list(train_df_balanced['question2'])

# Initialize a CountVectorizer object
vectorizer = CountVectorizer()

# Fit the CountVectorizer object to the list of questions
X = vectorizer.fit_transform(all_questions)

# Get the feature names and count the total number of features
features = vectorizer.vocabulary_
total_features = len(features)

print("Total number of features:", total_features)

print('-------------------------------------------\n')

# Combine all the questions into a single list
all_questions = list(
    train_df_balanced['question1']) + list(train_df_balanced['question2'])

# Initialize a CountVectorizer object
vectorizer = CountVectorizer(max_features=3000)

# Fit the CountVectorizer object to the list of questions
vectorizer.fit(all_questions)

# Transform individual questions into sparse matrices
q1_sparse = csr_matrix(vectorizer.transform(
    list(train_df_balanced['question1'])), dtype=np.int8)
q2_sparse = csr_matrix(vectorizer.transform(
    list(train_df_balanced['question2'])), dtype=np.int8)

# Combine the sparse matrices horizontally
combined_sparse = hstack([q1_sparse, q2_sparse])

# Convert the combined sparse matrix to a dataframe
combined_df = pd.DataFrame.sparse.from_spmatrix(combined_sparse)

# Free up memory used by q1_vector and q2_vector
del q1_sparse, q2_sparse

feature_names = list(vectorizer.vocabulary_.keys())
col_names = feature_names + feature_names

# Drop 'question1' and 'question2' columns from train_df_balanced
train_df_new = train_df_balanced.drop(['question1', 'question2'], axis=1)

# Concatenate train_df_new and combined_df dataframes
new_df = pd.concat([train_df_new, combined_df], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    new_df.iloc[:, 1:], new_df.iloc[:, 0], test_size=0.2, random_state=101)


# Save train and test sets
X_train.to_pickle('dataset/X_train.pkl')
X_test.to_pickle('dataset/X_test.pkl')
y_train.to_pickle('dataset/y_train.pkl')
y_test.to_pickle('dataset/y_test.pkl')
