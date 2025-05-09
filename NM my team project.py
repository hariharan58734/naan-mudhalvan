# Step 1: Install required packages (run this once)
# pip install pandas scikit-learn nltk matplotlib wordcloud

# Step 2: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string

# Step 3: Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 4: Load your data
# You can replace this with your own CSV file with columns: 'text' and 'label'
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_or_real_news.csv')
df = df[['text', 'label']]  # Keep only relevant columns

# Step 5: Preprocessing function
def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join([c for c in text if c not in string.punctuation])  # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return ' '.join(words)

# Step 6: Apply cleaning
df['text'] = df['text'].apply(clean_text)

# Step 7: Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 9: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 10: Make predictions and evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 11: Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
