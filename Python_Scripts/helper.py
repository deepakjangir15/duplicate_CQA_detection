from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
from nltk.stem import WordNetLemmatizer
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import word_tokenize
import distance
from fuzzywuzzy import fuzz


nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def get_missing_values(df):
    for col in df.columns:
        print("Column " + col + ' -- ' + str(sum(df[col].isna())))


def preprocess_data(q):
    q = str(q).lower().strip()

    # Create a dictionary of special characters and their string equivalents
    special_chars = {
        '%': 'percent',
        '$': 'dollar',
        '₹': 'rupee',
        '€': 'euro',
        '@': 'at'
    }

    # Replace special characters with their string equivalents using dictionary comprehension
    q = ''.join([special_chars.get(c, c) for c in q])

    # Replace '[math]' with empty string
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Fixing all the contractions present in the questions
    q = contractions.fix(q)

    # Removing HTML tags
    q = BeautifulSoup(q, features="html.parser")
    q = q.get_text()

    # Lemmatize the words in the question
    lemmatizer = WordNetLemmatizer()
    q = " ".join([lemmatizer.lemmatize(word) for word in q.split()])

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q


def pre_process_stack(text):
    # Define regex patterns
    regex_patterns = [
        r'<[^>]+>',  # HTML tags
        r'@(\w+)',  # @-mentions
        r"#(\w+)",  # hashtags
        # URLs
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-f][0-9a-f]))+',
        r'[^0-9a-z #+_\r\n\t]',  # BAD SYMBOLS
    ]

    # Define replacements for each regex pattern
    replacements = [
        ' ',  # HTML tags
        '',  # @-mentions
        '',  # hashtags
        '',  # URLs
        ' ',  # BAD SYMBOLS
    ]

    # Define stopwords and lemmatizer
    STOP_WORDS = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Function to get part of speech for lemmatization
    def get_pos_tag(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    # Define regex patterns and replacements for URL, hash and at mentions
    REPLACE_URLS = re.compile(
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*,]|(?:%[0-9a-f][0-9a-f]))+')
    REPLACE_HASH = re.compile(r'#(\w+)')
    REPLACE_AT = re.compile(r'@(\w+)')
    REPLACE_BY = re.compile(r"[^a-z0-9\-]")

    # Replace URLs, hash and at mentions
    text = REPLACE_URLS.sub('', text)
    text = REPLACE_HASH.sub('', text)
    text = REPLACE_AT.sub('', text)
    text = REPLACE_BY.sub(' ', text)

    # Replace HTML tags using BeautifulSoup
    soup = BeautifulSoup(text)
    for code_tag in soup.find_all('code'):
        code_tag.replace_with('CODE')
    text = soup.get_text().replace('\n', ' ')

    # Remove whitespace and hyphens from the text
    text = ' '.join(text.replace("-", "").split())

    # Remove all regex patterns and replace them with spaces
    for i in range(len(regex_patterns)):
        text = re.sub(regex_patterns[i], replacements[i], text)

    # Tokenize text and lemmatize each token
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(word.strip(), get_pos_tag(tag))
                         for word, tag in nltk.pos_tag(tokens)
                         if word not in STOP_WORDS and len(word) > 3]

    # Join the lemmatized tokens and return the pre-processed text
    return ' '.join(lemmatized_tokens)


# Function to remove tags from a string
def remove_tags(text):
    """
    Removes HTML tags from a string
    """
    return text.replace("<", "").replace(">", " ")[:-1]


FIXED_LEN_FOR_BODY = 300


def append_data(row, title, tags, body, body_len):
    final_len = min(int(body_len), FIXED_LEN_FOR_BODY)
    data = f"{title} {tags} {body[:final_len]}"
    return data


def append_original_data(row):
    return append_data(row, row['OTitle'], row['OTags'], row['OBody'], row['OBody_len'])


def append_duplicate_data(row):
    return append_data(row, row['DTitle'], row['DTags'], row['DBody'], row['DBody_len'])


def fetch_token_features(row):

    q1 = row['question1']
    q2 = row['question2']

    SAFE_DIV = 0.0001

    STOP_WORDS = stopwords.words("english")

    token_features = [0.0]*8

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features = [
        common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV),
        common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV),
        common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV),
        common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV),
        common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV),
        common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV),
        # Last word of both question is same or not
        int(q1_tokens[-1] == q2_tokens[-1]),
        # First word of both question is same or not
        int(q1_tokens[0] == q2_tokens[0])
    ]

    return token_features


def get_length_features(row):

    q1 = row['question1']
    q2 = row['question2']

    feature_length = [0.0]*3

    # Converting the Sentence into Tokens:
    q1_tokens = word_tokenize(q1)
    q2_tokens = word_tokenize(q2)

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return feature_length

    # Absolute length features
    feature_length[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Average Token Length of both Questions
    feature_length[1] = (len(q1_tokens) + len(q2_tokens))/2

    strs = list(distance.lcsubstrings(q1, q2))
    if len(strs) == 0:
        feature_length[2] = 0.0
    else:
        feature_length[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return feature_length


def get_length_features(row):

    q1 = row['question1']
    q2 = row['question2']

    feature_length = [0.0]*3

    # Converting the Sentence into Tokens:
    q1_tokens = word_tokenize(q1)
    q2_tokens = word_tokenize(q2)

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return feature_length

    # Absolute length features
    feature_length[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Average Token Length of both Questions
    feature_length[1] = (len(q1_tokens) + len(q2_tokens))/2

    strs = list(distance.lcsubstrings(q1, q2))
    if len(strs) == 0:
        feature_length[2] = 0.0
    else:
        feature_length[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return feature_length


def fetch_fuzzy_features(row):

    q1 = row['question1']
    q2 = row['question2']

    fuzzy_features = [0.0]*4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


def evaluate_all(y_test, y_pred):

    # Evaluate the model performance on the test data
    accuracy_lr = accuracy_score(y_test, y_pred)
    precision_lr = precision_score(y_test, y_pred)
    recall_lr = recall_score(y_test, y_pred)
    f1_lr = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy_lr}")
    print(f"Precision: {precision_lr}")
    print(f"Recall: {recall_lr}")
    print(f"F1 score: {f1_lr}")
