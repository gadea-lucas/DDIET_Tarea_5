# Standard library imports
import os
import re
import string

# Third party imports
from gensim.models import KeyedVectors
from nltk import SnowballStemmer, PorterStemmer, download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import zeros
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


""" Preparations """
# Download stopwords
download('punkt')
download('stopwords')
# Vectorizer TF and TF-IDF
tf_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
# Load the models
# NOTE: Set binary=False if your model is not in binary format
# English (Word2Vec: https://code.google.com/archive/p/word2vec/)
model_en = KeyedVectors.load_word2vec_format('./models/GoogleNews-vectors-negative300.bin', binary=True)
# Spanis (SBWCE: https://crscardellino.github.io/SBWCE/)
model_es = KeyedVectors.load_word2vec_format('./models/SBWCE_model.bin', binary=True) 

""" Functions """
def clean_html(content):
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', content) 
    # Optional: Remove common HTML entities. Expand as needed.
    clean_text = re.sub(r'&[^;]+;', ' ', clean_text)
    return clean_text

# Enhanced clean_text function
def clean_text(text, language='english', remove_html=False):
    if remove_html:
        text = clean_html(text)
    # Remove punctuation and make lowercase
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = text.translate(translator).lower()
    
    # Tokenize the cleaned text
    tokens = word_tokenize(cleaned_text, language=language)
    
    # Remove stop-words
    stop_words = set(stopwords.words(language))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer() if language == 'english' else SnowballStemmer('spanish')
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return ' '.join(stemmed_tokens)

def process_documents(folder_path, language='english', remove_html=False):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            processed_text = clean_text(text, language, remove_html)
            documents.append(processed_text)
    return documents

def document_vector_additive(model, doc):
    """Generate document vector by summing the vectors of all words in the document that are in the model."""
    document_vec = sum(model[word] for word in doc if word in model)
    return document_vec

def document_vector_average(model, doc):
    """Generate document vector by averaging the vectors of all words in the document that are in the model."""
    words_in_model = [model[word] for word in doc if word in model]
    if words_in_model:  # Check if there's at least one word found in the model
        document_vec = document_vector_sequential_combination(model, doc)
    else:
        document_vec = None  # Or handle the case where no words are found in the model
    return document_vec

def document_vector_sequential_combination(model, tokens):
    """Generate document vector by averaging the vectors of all words in the document that are in the model."""
    if not tokens:
        return None
    
    combined_vector = None
    for word in tokens:
        if word in model:
            if combined_vector is None:
                combined_vector = model[word]
            else:
                # Combine the current combined vector with the next word vector
                combined_vector = (combined_vector + model[word]) / 2  # Averaging the vectors

    return combined_vector

# Finally, the generated representations must be exported in text files.
def export_vectors_to_file(vectors, file_path):
    if not os.path.exists(file_path):
        # Si el archivo no existe, lo creamos
        mode = 'w'
    else:
        # Si ya existe añadimos
        mode = 'a'

    with open(file_path, mode) as f:
        for vector in vectors:
            if vector is not None:  # Check if the vector is not None
                vector_str = ' '.join(map(str, vector))  # Convert each vector element to string and join with spaces
                f.write(vector_str + "\n")
            else:
                f.write("[WARNING] Vector not available\n")  # Placeholder text or handle as appropriate


""" Execute program """
def main():
    # Process the documents from each collection
    # # Collection 1
    # tweets_en = process_documents('./colecciones/tweets_en', 'english', False)
    # # Collection 2
    # tweets_es = process_documents('./colecciones/tweets_es', 'spanish', False)
    # Collection 3
    web_en = process_documents('./colecciones/web_en', 'english', True)
    # Collection 4
    web_es = process_documents('./colecciones/web_es', 'spanish', True)

    # TF representation
    web_tf_en = tf_vectorizer.fit_transform(web_en)
    export_vectors_to_file(web_tf_en, './results/tf_web_en.txt')
    web_tf_es = tf_vectorizer.fit_transform(web_es)
    export_vectors_to_file(web_tf_es, './results/tf_web_es.txt')
    # tweet_tf_es = tf_vectorizer.fit_transform(tweets_es)
    # export_vectors_to_file(tweet_tf_es, './results/tf_tweets_es.txt')
    # tweet_tf_en = tf_vectorizer.fit_transform(tweets_en)
    # export_vectors_to_file(tweet_tf_en, './results/tf_tweets_en.txt')
    
    # TF representation
    web_tfidf_en = tfidf_vectorizer.fit_transform(web_en)
    export_vectors_to_file(web_tfidf_en, './results/tfidf_web_en.txt')
    web_tfidf_es = tfidf_vectorizer.fit_transform(web_es)
    export_vectors_to_file(web_tfidf_es, './results/tfidf_web_es.txt')
    # tweet_tfidf_en = tfidf_vectorizer.fit_transform(tweets_en)
    # export_vectors_to_file(tweet_tfidf_en, './results/tfidf_tweets_en.txt')
    # tweet_tfidf_es = tfidf_vectorizer.fit_transform(tweets_es)
    # export_vectors_to_file(tweet_tfidf_es, './results/tfidf_tweets_es.txt')


    # Semantic Vector Representations
    for doc in web_en:
        # Tokenize document
        tokens = doc.split()  # Assuming `clean_text` returns a space-separated string of tokens
        # Generate vectors
        additive_vector = document_vector_additive(model_en, tokens)
        average_vector = document_vector_average(model_en, tokens)
        # Export vectors
        export_vectors_to_file([additive_vector], './results/additive_web_en.txt')
        export_vectors_to_file([average_vector], './results/average_web_en.txt')
    
    for doc in web_es:
        # Tokenize document
        tokens = doc.split()  # Assuming `clean_text` returns a space-separated string of tokens
        # Generate vectors
        additive_vector = document_vector_additive(model_es, tokens)
        average_vector = document_vector_average(model_es, tokens)
        # Export vectors
        export_vectors_to_file([additive_vector], './results/additive_web_es.txt')
        export_vectors_to_file([average_vector], './results/average_web_es.txt')

    # for doc in tweets_en:
    #     # Tokenize document
    #     tokens = doc.split()  # Assuming `clean_text` returns a space-separated string of tokens
    #     # Generate vectors
    #     additive_vector = document_vector_additive(model_en, tokens)
    #     average_vector = document_vector_average(model_en, tokens)
    #     # Export vectors
    #     export_vectors_to_file([additive_vector], './results/additive_tweets_en.txt')
    #     export_vectors_to_file([average_vector], './results/average_tweets_en.txt')

    # for doc in tweets_es:
    #     # Tokenize document
    #     tokens = doc.split()  # Assuming `clean_text` returns a space-separated string of tokens
    #     # Generate vectors
    #     additive_vector = document_vector_additive(model_es, tokens)
    #     average_vector = document_vector_average(model_es, tokens)
    #     # Export vectors
    #     export_vectors_to_file([additive_vector], './results/additive_tweets_es.txt')
    #     export_vectors_to_file([average_vector], './results/average_tweets_es.txt')



if __name__ == "__main__":
    main()




