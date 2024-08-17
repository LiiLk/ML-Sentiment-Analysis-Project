from data_preprocessing import processed_documents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Vectorise data using the Term Frequency and Inverse Document Frequency
# Text will be convert into numerical vectors for be read by a model

documents_text = [' '.join(doc) for doc, category in processed_documents]
tfIdf = TfidfVectorizer(max_features=5000) # Import the TF-IDF method
X = tfIdf.fit_transform(documents_text) # our vectors data with TF-IDF
y = [category for doc, category in processed_documents] # labels with pos and neg for each reviews

# Divide data
# Using in input, the vectorise data using the TF-IDF and the labels
# X_train and y_train using 80% of data for train data
# the X_test and y_test will be the test after train the data. The test takes the 20% of data left(that's why text_size=0.2).
# Using 27 for random_state because why not but the parameters use an constant value to have the same divide of datas between training and the test.
# This guarantees that we can reproduce the results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# Create Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction and results give by the Model
y_pred = model.predict(X_test)
print(f"Précision : {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

