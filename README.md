<img width="1918" height="736" alt="image" src="https://github.com/user-attachments/assets/73a4ac3b-ecb0-4aba-aeef-5c9eb18e30b7" /># AI-Chatbot
## Introduction
This project implements a conversational AI chatbot that understands user messages, classifies their intent using NLP + Machine Learning, and returns appropriate responses based on a predefined knowledge base. It is designed to be lightweight, modular, and extendable.
## Architecture Overview
The project integrates several technologies to deliver an end-to-end conversational AI system:

**1. Python –** Core programming language for building the chatbot.

**2. Natural Language Toolkit (NLTK) –** For preprocessing and understanding user text.

**3. Deep Learning Models (TensorFlow/PyTorch) –** To power intent classification and response generation.

**4. Flask – **To serve the chatbot as a web application.

**5. HTML/CSS/JavaScript –** For a simple interactive UI.

![WhatsApp Image 2025-10-11 at 21 55 56_b8cd7a90](https://github.com/user-attachments/assets/7d57e8ff-b93a-4637-bb9f-6691a6aa6e58)

Architecture Flow:

**1.** User interacts with the chatbot through a web interface.

**2.** User text is preprocessed (tokenization, stemming/lemmatization, stop-word removal).

**3.** Intent classification model predicts the category of the query.

**4.** The chatbot retrieves or generates the appropriate response.

**5.** Response is displayed back to the user in the chat interface.

# Data Flow

**1. Data Source –** Predefined intents JSON file containing patterns and responses.

**2. Preprocessing –** Text is cleaned, tokenized, and converted into numerical form (Bag of Words/TF-IDF/Embeddings).

**3. Model Training –** A neural network is trained to classify intents.

**5. Response Generation –** Based on predicted intent, a matching response is returned.

**6. Deployment –** The chatbot is deployed via Flask for web-based interaction.

<img width="1918" height="736" alt="image" src="https://github.com/user-attachments/assets/78501ade-8221-43e0-bb3f-64d90a8aa9b2" />

