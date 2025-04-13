import json
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import random

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

class Chatbot:
    def __init__(self):
        try:
            with open('data/intents.json', 'r') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            print("Error: 'data/intents.json' not found.")
            return
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in 'intents.json'.")
            return

        text_data = []
        labels = []
        stopwords = set(nltk.corpus.stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        limit_per_tag = 40

        for intent in self.data['intents']:
            augmented_sentences_per_tag = 0
            for example in intent['patterns']:
                tokens = nltk.word_tokenize(example.lower())
                filtered_tokens = [
                    lemmatizer.lemmatize(token)
                    for token in tokens if token not in stopwords and token.isalpha()
                ]
                if filtered_tokens:
                    text_data.append(' '.join(filtered_tokens))
                    labels.append(intent['tag'])
                    augmented_sentences = self.synonym_replacement(filtered_tokens, limit_per_tag - augmented_sentences_per_tag)
                    for augmented_sentence in augmented_sentences:
                        text_data.append(augmented_sentence)
                        labels.append(intent['tag'])
                        augmented_sentences_per_tag += 1
                        if augmented_sentences_per_tag >= limit_per_tag:
                            break    

        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(text_data)
        y = labels

        self.best_model = self.find_best_model(X, y)

    def synonym_replacement(self, tokens, limit):
        augmented_sentences = []
        for i in range(len(tokens)):
            synonyms = []
            for syn in wordnet.synsets(tokens[i]):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            if synonyms:
                num_augmentations = min(limit, len(synonyms))
                sampled_synonyms = random.sample(synonyms, num_augmentations)
                for synonym in sampled_synonyms:
                    augmented_tokens = tokens[:i] + [synonym] + tokens[i+1:]
                    augmented_sentences.append(' '.join(augmented_tokens))
        return augmented_sentences

    def find_best_model(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=100)
        models = [
            ('Logistic Regression', LogisticRegression(), {
                'penalty': ['l2'],
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear'],
                'max_iter': [100, 1000, 10000]
            }),
            ('Decision Tree', DecisionTreeClassifier(), {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }),
            ('Random Forest', RandomForestClassifier(), {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            })
        ]

        best_score = 0
        best_model = None

        for name, model, param_grid in models:
            grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(f'{name}: {score:.4f} (best parameters: {grid.best_params_})')
            if score > best_score:
                best_score = score
                best_model = grid.best_estimator_

        print(f'\nBest model: {best_model}')
        best_model.fit(X, y)

        return best_model
    

    def start_chatbot(self, user_input):
        input_text = self.vectorizer.transform([user_input])
        predicted_intent = self.best_model.predict(input_text)[0]
    
        for intent in self.data['intents']:
            if intent['tag'] == predicted_intent:
                response = random.choice(intent['responses'])
                return f"{response}"
