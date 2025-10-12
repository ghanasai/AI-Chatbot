# AI-Chatbot

## Introduction
This project implements an intelligent chatbot capable of responding to user queries using Natural Language Processing (NLP) and Machine Learning. The chatbot is designed to simulate human-like conversation, offering a scalable solution for customer support, information retrieval, or educational assistance.

---

## Architecture Overview
The project integrates several technologies to deliver an end-to-end conversational AI system:

- **Python** – Core programming language for building the chatbot.  
- **Natural Language Toolkit (NLTK) / spaCy** – For preprocessing and understanding user text.  
- **Deep Learning Models (TensorFlow / PyTorch)** – To power intent classification and response generation.  
- **Flask** – To serve the chatbot as a web application.  
- **HTML / CSS / JavaScript** – For a simple interactive UI.

  ![WhatsApp Image 2025-10-11 at 21 55 56_3447bd50](https://github.com/user-attachments/assets/578dda16-6fdd-467e-9c2e-596ef6ba9279)


### Architecture Flow
1. User interacts with the chatbot through a web interface.  
2. User text is preprocessed (tokenization, stemming / lemmatization, stop-word removal).  
3. Intent classification model predicts the category of the query.  
4. The chatbot retrieves or generates the appropriate response.  
5. Response is displayed back to the user in the chat interface.

---

## Data Flow
1. **Data Source** – Predefined intents JSON file containing patterns and responses.  
2. **Preprocessing** – Text is cleaned, tokenized, and converted into numerical form (Bag of Words / TF-IDF / Embeddings).  
3. **Model Training** – A neural network is trained to classify intents.  
4. **Response Generation** – Based on predicted intent, a matching response is returned.  
5. **Deployment** – The chatbot is deployed via Flask for web-based interaction.

<img width="1919" height="736" alt="Screenshot 2025-10-12 192322" src="https://github.com/user-attachments/assets/68907175-0bab-4e69-949a-39903e8367be" />


---

## Dataset
The project uses a custom intents dataset (in JSON format) that defines:

- **Intents** – Categories of user queries (e.g., greetings, farewells, FAQs).  
- **Patterns** – Example user inputs for each intent.  
- **Responses** – Possible chatbot replies.

---

## Key Steps
- **Data Preparation** – Build intents dataset with patterns and responses.  
- **Text Preprocessing** – Tokenization, lemmatization, and vectorization.  
- **Model Training** – Train an NLP / ML model for intent classification.  
- **Chatbot Engine** – Develop logic to map classified intent to responses.  
- **Web Integration** – Use Flask + HTML / CSS / JS for the frontend chat interface.

---

## Code Explanation

### Libraries and Data
This project begins by importing the necessary libraries:

- **JSON** – For reading and handling the intents data.  
- **NLTK** – For text preprocessing, tokenization, and lemmatization.  
- **Scikit-learn** – For machine learning, model training, and evaluation.  
- **Flask** – For building and deploying the web application.

The project reads an `intents.json` file that contains example user queries (patterns) and possible replies (responses) for each intent.  
For example, an intent such as `"greeting"` includes patterns like `"hi"` or `"hello"` and responses like `"Hello! How can I assist you?"`:contentReference[oaicite:0]{index=0}.

<img width="1166" height="647" alt="image" src="https://github.com/user-attachments/assets/cebdc376-1a43-4a53-932b-3c246b230bf4" />


---

### Text Preprocessing and Data Augmentation
The input queries undergo several NLP preprocessing steps:

- **Tokenization:** Splits sentences into individual words.  
- **Stopword Removal:** Eliminates common words (e.g., “the”, “a”, “is”) that add little meaning.  
- **Lemmatization:** Reduces words to their base form (e.g., “Running” → “Run”).  
- **Synonym Replacement:** Expands the dataset by replacing words with synonyms from WordNet to improve model robustness and generalization:contentReference[oaicite:1]{index=1}.

<img width="1162" height="577" alt="image" src="https://github.com/user-attachments/assets/fddff99e-2d62-47b1-97b2-d0efaa464d92" />

<img width="1161" height="496" alt="image" src="https://github.com/user-attachments/assets/0147b35c-fc5b-4703-bbe3-c83a6408e0bc" />

<img width="1162" height="447" alt="image" src="https://github.com/user-attachments/assets/ba26459b-75b1-4444-93ce-78df45ab3f75" />

---

### Model Training and Evaluation
After preprocessing, the text is converted into numerical form using **TF-IDF vectorization**.  
Several machine-learning algorithms are trained and compared:

- **Logistic Regression**  
- **Decision Tree**  
- **Random Forest**

<img width="1156" height="730" alt="image" src="https://github.com/user-attachments/assets/2bfb57da-cf20-4c79-90d6-dc3477701c34" />


A **Grid Search Cross-Validation** procedure is applied to tune hyperparameters and select the best-performing model based on predictive accuracy.  
The dataset is split into 80% training and 20% testing portions to evaluate generalization:contentReference[oaicite:2]{index=2}.

---

### Web Interface using Flask
The chatbot is integrated with a **Flask web interface**, which handles user interaction through HTTP requests.

- When a user types a message in the browser, Flask routes it to the chatbot model for intent prediction.

<img width="1162" height="532" alt="image" src="https://github.com/user-attachments/assets/cfcd291a-a480-408b-9e2e-d7bf07dc56c6" />

- Flask then retrieves the appropriate response from `intents.json` and returns it to the frontend chat window.  
- The `Markup` function is used to properly format HTML responses on the web page:contentReference[oaicite:3]{index=3}.

<img width="1160" height="587" alt="image" src="https://github.com/user-attachments/assets/ca029c93-c140-4b5d-a7bd-29f07a1b9b52" />

---

### Chatbot Response Handling
1. User input is received through the web interface.  
2. Flask passes the query to the trained model for intent prediction.  
3. The predicted intent is used to look up responses in `intents.json`.  
4. The selected response is displayed to the user through the chat window in real-time:contentReference[oaicite:4]{index=4}.

<img width="1162" height="531" alt="image" src="https://github.com/user-attachments/assets/de7927e0-134f-4dc5-96c8-5341e8e413c7" />

<img width="1161" height="622" alt="image" src="https://github.com/user-attachments/assets/98596665-f437-4553-a991-924365ca6233" />

<img width="1160" height="613" alt="image" src="https://github.com/user-attachments/assets/de49337b-02ae-4096-8281-88aca0d447d1" />

---

## Insights
- **User Engagement** – The chatbot provides real-time automated responses.  
- **Intent Accuracy** – With proper training, the intent classifier achieves high accuracy in understanding queries.  
- **Scalability** – The system can be extended with APIs (e.g., weather, news, knowledge base).

---

## Technologies Used
- **Programming**: Python  
- **Libraries**: NLTK, spaCy, Scikit-learn, TensorFlow / PyTorch  
- **Framework**: Flask  
- **Frontend**: HTML, CSS, JavaScript  

---

## Future Work
- Improve chatbot accuracy using transformer models (BERT / GPT).  
- Add voice recognition (speech-to-text) support.  
- Integrate with messaging platforms (Slack, WhatsApp, Telegram).  
- Add a knowledge base retrieval system for FAQ automation.
