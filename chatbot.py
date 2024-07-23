import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import random
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import re

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
def preprocess_sentence(sentence):
    sentence = re.sub(r"([.!?])", r" \1", sentence)
    sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
    sentence = sentence.lower().strip()
    return sentence

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.lower() for token in word_tokenize(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

class ChatDataset(Dataset):
    def __init__(self, X, y, source_vocab, target_vocab):
        self.X = X
        self.y = y
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        source = torch.tensor(self.source_vocab.numericalize(self.X[index]))
        target = torch.tensor(self.target_vocab.numericalize(self.y[index]))
        return source, target

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens

# Seq2Seq model with attention
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        prediction = self.out(output.squeeze(1))
        return prediction, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)
        encoder_output, hidden = self.encoder(source)

        input = target[:, 0]

        for t in range(1, target_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_output)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force else top1

        return outputs

# Training function
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg, _, _) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg, _, _) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Prediction function
def predict(model, sentence, source_vocab, target_vocab, max_length=50):
    model.eval()
    tokens = source_vocab.tokenize(preprocess_sentence(sentence))
    tokens = ['<SOS>'] + tokens + ['<EOS>']
    src_indexes = [source_vocab.stoi.get(token, source_vocab.stoi['<UNK>']) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    encoder_outputs, hidden = model.encoder(src_tensor)
    
    trg_indexes = [target_vocab.stoi['<SOS>']]
    
    for i in range(max_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == target_vocab.stoi['<EOS>']:
            break
    
    trg_tokens = [target_vocab.itos[i] for i in trg_indexes]
    return ' '.join(trg_tokens[1:-1])  # remove <SOS> and <EOS>

# Main execution
if __name__ == "__main__":
    # Sample data (replace with your own dataset)
    pairs = [
        ("hello", "hi there"),
        ("how are you", "i'm doing well, thanks"),
        ("what's your name", "i'm a chatbot"),
        ("bye", "goodbye"),
        ("tell me a joke", "why did the chicken cross the road? to get to the other side!"),
        ("what's the weather like", "i'm sorry, i don't have real-time weather information"),
        ("do you like music", "i enjoy all kinds of music, what's your favorite?"),
        ("what's your favorite color", "as an AI, i don't have personal preferences, but i find all colors fascinating"),
        ("can you help me with my homework", "i'd be happy to try! what subject are you working on?"),
        ("what's the meaning of life", "that's a deep question! philosophers have debated it for centuries"),
    ]

    # Preprocess data
    source_sentences = [preprocess_sentence(pair[0]) for pair in pairs]
    target_sentences = [preprocess_sentence(pair[1]) for pair in pairs]

    # Create vocabularies
    source_vocab = Vocabulary(freq_threshold=1)
    target_vocab = Vocabulary(freq_threshold=1)

    source_vocab.build_vocabulary(source_sentences)
    target_vocab.build_vocabulary(target_sentences)

    # Create dataset and dataloader
    dataset = ChatDataset(source_sentences, target_sentences, source_vocab, target_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_collate)

    # Initialize the model
    INPUT_DIM = len(source_vocab)
    OUTPUT_DIM = len(target_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = EncoderRNN(INPUT_DIM, HID_DIM, ENC_DROPOUT).to(device)
    dec = DecoderRNN(HID_DIM, OUTPUT_DIM, DEC_DROPOUT).to(device)

    model = Seq2Seq(enc, dec).to(device)

    # Training parameters
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=source_vocab.stoi["<PAD>"])
    CLIP = 1
    N_EPOCHS = 100

    # Training loop
    for epoch in range(N_EPOCHS):
        train_loss = train(model, dataloader, optimizer, criterion, CLIP)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')
        
    # Save the model
    torch.save(model.state_dict(), 'chatbot_model.pth')
    
    # Save the vocabularies
    torch.save({
        'source_vocab': source_vocab,
        'target_vocab': target_vocab
    }, 'vocab.pth')

    print("Model and vocabularies saved.")

    # Chat loop
    print("Chat with the bot (type 'quit' to exit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = predict(model, user_input, source_vocab, target_vocab)
        print("Chatbot:", response)