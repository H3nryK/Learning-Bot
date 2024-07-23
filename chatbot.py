import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import torch.optim as optim
import random

# Define the encoder (no changes)
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

# Define the decoder with attention
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Define the chatbot (no changes)
class Chatbot:
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def train(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion, max_length=50):
        encoder_hidden = torch.zeros(1, 1, self.encoder.hidden_size, device=self.device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[0]], device=self.device)
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))  # Add unsqueeze here
            decoder_input = target_tensor[di]  # Teacher forcing

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def predict(self, input_tensor, max_length=50):
        with torch.no_grad():
            encoder_hidden = torch.zeros(1, 1, self.encoder.hidden_size, device=self.device)

            input_length = input_tensor.size(0)
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)

            decoder_input = torch.tensor([[0]], device=self.device)
            decoder_hidden = encoder_hidden

            decoded_words = []

            for di in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                if topi.item() == 1:
                    break
                else:
                    decoded_words.append(topi.item())

                decoder_input = topi.squeeze().detach()

            return decoded_words

# Helper functions for data preparation and training
def prepare_data(pairs, input_lang, output_lang):
    input_tensors = [torch.tensor([input_lang.word2index[word] for word in pair[0].split()], dtype=torch.long).to(device) for pair in pairs]
    target_tensors = [torch.tensor([output_lang.word2index[word] for word in pair[1].split()], dtype=torch.long).to(device) for pair in pairs]
    return input_tensors, target_tensors

def train_chatbot(chatbot, pairs, input_lang, output_lang, n_epochs, learning_rate=0.01):
    encoder_optimizer = optim.SGD(chatbot.encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(chatbot.decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    input_tensors, target_tensors = prepare_data(pairs, input_lang, output_lang)

    for epoch in range(n_epochs):
        total_loss = 0
        for input_tensor, target_tensor in zip(input_tensors, target_tensors):
            loss = chatbot.train(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion)
            total_loss += loss

        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(pairs):.4f}')

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample data (you should replace this with your own dataset)
    pairs = [
        ("hello", "hi there"),
        ("how are you", "i'm doing well, thanks"),
        ("what's your name", "i'm a chatbot"),
        ("bye", "goodbye")
    ]

    # Create vocabulary
    input_words = set(word for pair in pairs for word in pair[0].split())
    output_words = set(word for pair in pairs for word in pair[1].split())

    # Add special tokens
    input_words.add('<SOS>')
    input_words.add('<EOS>')
    output_words.add('<SOS>')
    output_words.add('<EOS>')

    input_lang = type('Lang', (), {'word2index': {word: i for i, word in enumerate(input_words)}, 'index2word': {i: word for i, word in enumerate(input_words)}})()
    output_lang = type('Lang', (), {'word2index': {word: i for i, word in enumerate(output_words)}, 'index2word': {i: word for i, word in enumerate(output_words)}})()

    # Initialize the model
    hidden_size = 256
    encoder = Encoder(len(input_words), hidden_size).to(device)
    decoder = Decoder(hidden_size, len(output_words)).to(device)
    chatbot = Chatbot(encoder, decoder, device)

    # Train the model
    train_chatbot(chatbot, pairs, input_lang, output_lang, n_epochs=100)

    # Test the chatbot
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        input_tensor = torch.tensor([input_lang.word2index.get(word, 0) for word in user_input.split()], dtype=torch.long).unsqueeze(1).to(device)
        output_indices = chatbot.predict(input_tensor)
        response = ' '.join([output_lang.index2word.get(index, "<UNK>") for index in output_indices])
        print("Chatbot:", response)