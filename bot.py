import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense

class Chatbot:
    def __init__(self):
        self.responses = {}
        self.tokenizer = Tokenizer()
        self.model = None
        
    def prepare_data(self, texts):
        self.tokenizer.fit_on_texts(texts)
        
    def build_model(self):
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = Sequential([
            Embedding(vocab_size, 64),
            GlobalAveragePooling1D(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        
    def train_model(self, input_texts, labels):
        sequences = self.tokenizer.texts_to_sequences(input_texts)
        padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')
        labels = np.array(labels)
        self.model.fit(padded_sequences, labels, epochs=10)
    
    def generate_response(self, input_text):
        sequence = self.tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=10, padding='post')
        prediction = self.model.predict(padded_sequence)[0][0]
        if prediction >= 0.5:
            return "Yes"
        else:
            return "No"
        
    def learn(self, input_text, feedback):
        self.responses[input_text] = feedback

    def chat(self):
        print("Chatbot: Hello! how can I assist you today?")
        while True:
            user_input = input("User: ").lower()
            if user_input == 'exit':
                print("Chatbot: Goodbye!")
                break
            if user_input in self.responses:
                response = self.responses[user_input]
                print("Chatbot: ", response)
            else:
                print("Chatbot: I'm not sure how to respond. Can you please teach me?")
                feedback = input("You: ")
                self.learn(user_input, feedback)
                print("Chatbot: Thank you for teaching me.")

# Additional diverse and relevant training data
training_data = {
    "hi": 1,
    "how are you": 1,
    "bye": 1,
    "what's your name?": 1,
    "tell me a joke": 0,
    "what's the weather like today?": 0,
    "can you recommend a good book?": 0,
    "how do I bake a cake?": 0
}

# Initialize the chatbot and train model
chatbot = Chatbot()
chatbot.prepare_data(training_data.keys())
chatbot.build_model()

# Extract input_texts and labels from training_data
input_texts = list(training_data.keys())
labels = list(training_data.values())

# Train the model with the modified training data
chatbot.train_model(input_texts, labels)

# Interact with the chatbot
chatbot.chat()
