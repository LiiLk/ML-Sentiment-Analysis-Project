import random
import string
import nltk
from nltk.corpus import movie_reviews

#Download with nltk the dataset for movie reviews
nltk.download('movie_reviews')
nltk.download('punkt')

# Load the reviews movies
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to have an random sample
random.shuffle(documents)

# Preprocessing of text with cleaning and tokenisation
def preprocess_review(review): 
    review = ' '.join(review) # Seperate word
    review = review.lower()
    review = review.translate(str.maketrans('', '', string.punctuation)) # Deleting the punctuation
    words = nltk.word_tokenize(review)
    return words

# Apply the preprocessing
processed_documents = [(preprocess_review(doc), category) for doc, category in documents]

print(f"Premier exemple prétraité : {processed_documents[0]}")


