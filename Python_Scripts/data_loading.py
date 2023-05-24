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

# -----> CHAPTER 3 Loading the datasets

train_df = pd.read_csv("dataset/train.csv")

stack_data_dup = pd.read_csv(
    'dataset/Stack Overflow/SO_duplicate.csv')
stack_data_non_dup_p1 = pd.read_csv(
    'dataset/Stack Overflow/SO_Part1.csv')
stack_data_non_dup_p2 = pd.read_csv(
    'dataset/Stack Overflow/SO_Part2.csv')

stack_data_non_dup = pd.concat(
    [stack_data_non_dup_p1, stack_data_non_dup_p2], axis=1)

stack_data = pd.concat([stack_data_non_dup, stack_data_dup]
                       ).sample(frac=1, random_state=101)


stack_data_non_dup = pd.concat(
    [stack_data_non_dup_p1, stack_data_non_dup_p2], axis=1)


# -----> CHAPTER 4 Data Exploration & Preprocessing - Quora

# Create the visualization folder if it doesn't exist
folder_path = 'visualizations'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Plot the chart
ax = train_df['is_duplicate'].value_counts().plot(kind='bar', figsize=(6, 4))
x_labels = ['non-duplicate', 'duplicate']

ax.set_xlabel('Duplicate question distribution for Quora')
ax.set_ylabel('No of Questions')
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_xticklabels(x_labels)

# Save the image in the folder
image_path = os.path.join(folder_path, 'duplicate_question_distribution.png')
plt.savefig(image_path)

# Clear the current figure
plt.clf()

print('Missing values present in the Quora Dataset\n')
helper.get_missing_values(train_df)
print('-------------------------------------------\n')

# Dropping the rows that have missing values
train_df = train_df.dropna()

print('Missing values present in the Quora Dataset\n')
train_df['is_duplicate'].value_counts()
print('-------------------------------------------\n')

# -----> CHAPTER 4.3.3 Balancing the dataset

# Separate the majority and minority class
df_majority = train_df[train_df['is_duplicate'] == 0]
df_minority = train_df[train_df['is_duplicate'] == 1]

# Undersample the majority class
df_majority_undersampled = df_majority.sample(
    n=len(df_minority), random_state=101)

# Concatenate the majority undersampled class and minority class
train_df_balanced = pd.concat([df_majority_undersampled, df_minority])

# Shuffle the dataset
train_df_balanced = train_df_balanced.sample(frac=1, random_state=101)
print(train_df_balanced['is_duplicate'].value_counts())

print('-------------------------------------------\n')

ax = train_df_balanced.is_duplicate.value_counts().plot(kind='bar')
ax.bar_label(ax.containers[0], label_type='edge')

# Save the image in the folder
image_path = os.path.join(folder_path, 'balanced_dataset.png')
plt.savefig(image_path)

# Clear the current figure
plt.clf()

# -----> CHAPTER 4.4 Data Exploration

all_qid = pd.Series(train_df_balanced['qid1'].tolist(
) + train_df_balanced['qid2'].tolist())

all_qid_counts = all_qid.value_counts()

repeated_qns = all_qid_counts[all_qid_counts > 1].shape[0]
non_repeated_qns = all_qid_counts[all_qid_counts == 1].shape[0]

qns = [repeated_qns, non_repeated_qns]

# Plot a pie chart for the distribution

labels = ['Repeated Questions', 'Non-repeated Questions']
# Define the colors for the pie chart
colors = ['#ff9999', '#66b3ff']
# Create the pie chart
fig, ax = plt.subplots()
ax.pie(qns, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
# Set the axis aspect ratio to be equal, and add a title
ax.axis('equal')
plt.title('Distribution of Repeated vs. Non-repeated Questions')
# Save the image in the folder
image_path = os.path.join(
    folder_path, 'Distribution of Repeated vs. Non-repeated Questions.png')
plt.savefig(image_path)


plt.figure(figsize=(8, 4))
plt.hist(all_qid.value_counts().values, bins=150)
plt.yscale('log')
plt.title("Frequency of questions occuring")
# Save the image in the folder
image_path = os.path.join(
    folder_path, 'Frequency of questions occuring.png')
plt.savefig(image_path)

# -----> CHAPTER 4.5 Data Preprocessing Function for NLP Tasks

train_df_balanced['question1'] = train_df_balanced['question1'].apply(
    helper.preprocess_data)

train_df_balanced['question2'] = train_df_balanced['question2'].apply(
    helper.preprocess_data)


# -----> CHAPTER 5 Data Exploration & Preprocessing - Stack Overflow

ax = stack_data['isDuplicate'].value_counts().plot(kind='bar', figsize=(6, 4))
x_labels = ['non-duplicate', 'duplicate']

ax.set_xlabel('Duplicate question distribution for Stack Overflow')
ax.set_ylabel('No of Questions')
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_xticklabels(x_labels)
# Save the image in the folder
image_path = os.path.join(
    folder_path, 'Duplicate question distribution for Stack Overflow.png')
plt.savefig(image_path)
# Clear the current figure
plt.clf()


print('Missing values present in the StackOverflow Dataset\n')
helper.get_missing_values(stack_data)
print('-------------------------------------------\n')


# Applying the function to all the questions present in the original and duplicate body column in order to check well the cleaning processed

stack_data['OBody'] = stack_data['OBody'].apply(helper.pre_process_stack)
stack_data['DBody'] = stack_data['DBody'].apply(helper.pre_process_stack)

stack_data['OTags'] = stack_data.OTags.apply(helper.remove_tags)
stack_data['DTags'] = stack_data.DTags.apply(helper.remove_tags)

# -----> CHAPTER 5.3 Question Extraction

stack_data['OBody_len'] = stack_data['OBody'].str.len()
stack_data['DBody_len'] = stack_data['DBody'].str.len()


g = sns.displot(stack_data['OBody_len']).set(
    title='Analysis of the no of characters')
g.set(xlim=(0, 2000))
# Save the image in the folder
image_path = os.path.join(folder_path, 'Analysis of the no of characters.png')
plt.savefig(image_path)
# Clear the current figure
plt.clf()


# Merging the OTitle, OBody and OTags into a single column inorder to grab all the features

stack_data['question1'] = stack_data.apply(
    lambda x: helper.append_original_data(x), axis=1)
stack_data['question2'] = stack_data.apply(
    lambda x: helper.append_duplicate_data(x), axis=1)

stack_data['q1len'] = stack_data['question1'].str.len()
stack_data['q2len'] = stack_data['question2'].str.len()

stack_data = stack_data[['question1', 'question2', 'isDuplicate']]
quora_data = train_df_balanced[['question1', 'question2', 'is_duplicate']]

# renaming the column in the stack_data dataframe to match the name in the quora_data dataframe.
stack_data = stack_data.rename(columns={'isDuplicate': 'is_duplicate'})

train_df_balanced = pd.concat([stack_data[['question1', 'question2', 'is_duplicate']], quora_data[[
                              'question1', 'question2', 'is_duplicate']]], ignore_index=True)


train_df_balanced.to_pickle(
    'dataset/train_df_balanced.pkl')
