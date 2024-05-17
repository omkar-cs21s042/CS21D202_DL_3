from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import pandas a pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from matplotlib.font_manager import FontProperties as fontp

import argparse

import wandb

wandb.login(key='')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 0



CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--wandb_project",  
  nargs="*",  
  type=str,
  default="myproject",  
)
CLI.add_argument(
  "--wandb_entity",  
  nargs="*",  
  type=str,
  default="myname",  
)
CLI.add_argument(
  "--attention",  
  nargs="*",  
  type=bool,
  default=True,  
)
CLI.add_argument(
  "--epochs",
  nargs="*",
  type=int, 
  default=10,
)
CLI.add_argument(
  "--batchSize",
  nargs="*",
  type=int, 
  default=64,
)
CLI.add_argument(
  "--learningRate",
  nargs="*",
  type=float, 
  default=0.002,
)
CLI.add_argument(
  "--inputEmbedding",
  nargs="*",
  type=int, 
  default=32,
)
CLI.add_argument(
  "--outputEmbedding",
  nargs="*",
  type=int, 
  default=256,
)
CLI.add_argument(
  "--dropout",
  nargs="*",
  type=float, 
  default=0.1,
)
CLI.add_argument(
  "--cellType",
  nargs="*",
  type=str, 
  default="rnn",
)
CLI.add_argument(
  "--optimizer",
  nargs="*",
  type=str, 
  default="adam",
)
CLI.add_argument(
  "--layers",
  nargs="*",
  type=int, 
  default=[256, 128],
)
CLI.add_argument(
  "--beamWidth",
  nargs="*",
  type=int, 
  default=3,
)
CLI.add_argument(
  "--trainPath",
  nargs="*",
  type=str, 
  default="/content/dataset/train",
)
CLI.add_argument(
  "--valPath",
  nargs="*",
  type=str, 
  default="/content/dataset/train",
)
CLI.add_argument(
  "--testPath",
  nargs="*",
  type=str, 
  default="/content/dataset/val",
)

# parse the command line
args = CLI.parse_args()

sweep_config = {
  'name': 'Test_run',
  'method': 'bayes',
  'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
  'parameters': {
        'input_enbedding_size': {
            'values': [args.inputEmbedding]
        },
        'output_embedding_size': {
            'values': [args.outputEmbedding]
        },
        'cell_type': {
            'values': [args.cellType]
        },
        'encoder_decoder_layers': {
            'values': [args.layers]
        },
        'dropout': {
            'values': [args.dropout]
        },
        'optimiser': {
            'values': [args.optimizer]
        },
        'learning_rate': {
            'values': [args.learningRate]
        },
        'epochs': {
            'values': [args.epochs]
        },
        'beam_width': {
            'values': [args.beamWidth]
        },
        'batch_size':{
              'values': [args.batchSize]
        }
        'attention':{
              'values': [args.attention]
        }


    }
}







class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, inputEmbedding, cell_type, dropout):
        super(CustomRNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.cell_type = cell_type

        self.dropout = nn.Dropout(dropout)

        # Define the embedding layer
        self.embedding = nn.Embedding(input_size, inputEmbedding)
        
        # Define the RNN layers
        self.rnn_layers = nn.ModuleList()
        for i in range(len(self.hidden_sizes)):
            if i == 0:
                layer_input_size = inputEmbedding  # Input size matches the embedding size
            else:
                layer_input_size = self.hidden_sizes[i - 1]

            # adding layer based on cell type
            if self.cell_type == 'lstm':
                rnn_layer = nn.LSTM(layer_input_size, hidden_sizes[i], num_layers=1)
            elif self.cell_type == 'gru':
                rnn_layer = nn.GRU(layer_input_size, hidden_sizes[i], num_layers=1)
            else: 
                rnn_layer = nn.RNN(layer_input_size, hidden_sizes[i], num_layers=1)
            self.rnn_layers.append(rnn_layer)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output = embedded
        for i in range(len(self.hidden_sizes)):
            if i== 0:
                output, hidden = self.rnn_layers[i](output)        # no hidden state for first layer
            else:
                output, hidden = self.rnn_layers[i](output, hidden)   # forward pass through other layers
        return output, hidden
        


class BeamSearchNode:
    def __init__(self, hiddenstate, previousNode, letterId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.letterid = letterId
        self.logp = logProb
        self.length = length


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_sizes, outputEmbedding, cell_type, dropout, attention):
        super(AttnDecoderRNN, self).__init__()

        self.isAttnEnabled = attention

        self.embedding = nn.Embedding(output_size, outputEmbedding)

        if self.isAttnEnabled:
            self.attention = BahdanauAttention(outputEmbedding)
        
        # Define the RNN layers
        self.rnn_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                if self.isAttnEnabled:
                    layer_input_size = outputEmbedding * 2 # Input size 2 times because attention is concat
                else:
                    layer_input_size = outputEmbedding
            else:
                layer_input_size = hidden_sizes[i - 1]

            # adding layer based on cell type
            if self.cell_type == 'lstm':
                rnn_layer = nn.LSTM(layer_input_size, hidden_sizes[i], num_layers=1)
            elif self.cell_type == 'gru':
                rnn_layer = nn.GRU(layer_input_size, hidden_sizes[i], num_layers=1)
            else: 
                rnn_layer = nn.RNN(layer_input_size, hidden_sizes[i], num_layers=1)
            self.rnn_layers.append(rnn_layer)
        
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, beam_width):

        if target_tensor is not None:
            return self.training_forward(encoder_outputs, decoder_hidden, target_tensor)
        else:
            return self.inference_forward(encoder_outputs, encoder_hidden, beam_width)


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        # if attention is off
        output = embedded
        attn_weights = []

        if self.isAttnEnabled:
            query = hidden.permute(1, 0, 2)
            context, attn_weights = self.attention(query, encoder_outputs)
            output = torch.cat((embedded, context), dim=2)

        for i in range(len(self.hidden_sizes)):
            output, hidden = self.rnn_layers[i](output, hidden)   # forward pass through other layers

        output = self.out(output)

        return output, hidden, attn_weights


    def training_forward(encoder_outputs, encoder_hidden, target_tensor):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            if self.isAttnEnabled:
                attentions.append(attn_weights)

            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        if isAttnEnabled:
            attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def inference_forward(encoder_outputs, encoder_hidden, beam_width):
        topk = beam_width  # Beam width
        decoded_batch = []
        for b in range(encoder_outputs.size(0)):
            encoder_output = encoder_outputs[b].unsqueeze(0)
            decoder_hidden = encoder_hidden[:, b, :].unsqueeze(1)

            # Start with the initial input
            decoder_input = torch.LongTensor([[SOS_token]]).to(device)

            # Number of words to generate
            endnodes = []
            number_required = topk - len(endnodes)

            # Starting node -  hidden vector, previous node, letter id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # Start the queue
            nodes.put((-node.logp, node))
            qsize = 1

            # Start beam search
            while True:
                # Give up when decoding takes too long
                if qsize > 2000:
                    break

                # Fetch the best node
                score, n = nodes.get()
                decoder_input = n.letterid
                decoder_hidden = n.h

                if n.letterid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # If we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # Decode for one step using decoder
                decoder_output, decoder_hidden, _ = self.forward_step(decoder_input, decoder_hidden, encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, topk)
                nextnodes = []

                for new_k in range(topk):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.length + 1)
                    nextnodes.append((-node.logp, node))

                # Put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # Choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            words = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                word = []
                word.append(n.letterid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    word.append(n.letterid)

                word = word[::-1]
                if len(word) < MAX_LENGTH:
                    if word[-1] != EOS_token:
                        word.append(EOS_token)
                    while len(word) < MAX_LENGTH:
                        word.append(0)
                words.append(word)

            decoded_batch.append(words)

        return np.array(decoded_batch)


def read_data(path, inputLanguage, outputLanguage):
  pairs = []
  with open(path, "r", encoding="utf-8") as f:
      rows = f.read().split("\n")
  for row in rows[: len(rows) - 1]:
      input_text, target_text= row.split(",")
      pairs.append((input_text, target_text))
      inputLanguage.addWord(input_text)
      outputLanguage.addWord(output_text)
      if len(target_text) > MAX_LENGTH :
        MAX_LENGTH = len(target_text)

  return inputLanguage, outputLanguage, pairs



class Lang:
    def __init__(self, name):
        self.name = name
        self.letter2index = {}
        self.letter2count = {}
        self.index2letter = {0: "SOS", 1: "EOS"}
        self.n_letters = 2  # Count SOS and EOS

    def addWord(self, word):
        for letter in word:
            self.addLetter(word)

    def addLetter(self, letter):
        if letter not in self.letter2index:
            self.letter2index[letter] = self.n_letters
            self.letter2count[letter] = 1
            self.index2letter[self.n_letters] = letter
            self.n_letters += 1
        else:
            self.letter2count[letter] += 1

def indexesFromWord(lang, word):
    return [lang.letter2index[letter] for letter in word]

def wordFromIndexes(lang, word):
    return ''.join([lang.index2letter[index] for index in word])


def tensorFromWord(lang, word):
    indexes = indexesFromSentence(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def get_dataloader(batch_size, pairs, inputLanguage, outputLanguage):

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(inputLanguage, inp)
        tgt_ids = indexesFromSentence(outputLanguage, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()


        result_tensor = torch.argmax(decoder_outputs, dim=-1)
        target_array = target_tensor.numpy()
        result_array = result_tensor.numpy()
        total_correct += np.sum(np.all(target_array == result_array, axis=1))

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = total_correct/ len(dataloader)

    return epoch_loss, epoch_accuracy


def eval(encoder, decoder, dataloader, writeToCSV):
    global outCSV
    total_correct = 0
    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs = decoder(encoder_outputs, encoder_hidden, None)
            target_array = target_tensor.numpy()
            for i in range(len(target)):
                predictedWord = None
                if np.any(np.all(target[i] == decoder[i], axis=1)):
                    total_correct += 1
                    predictedWord = target[i][0]
                else:
                    predictedWord = decoder[i][0]
                if writeToCSV:
                    inputWord = inputTensor[i][0]
                    groundTruth = target[i][0]
                    inputWord = wordFromIndexes(inputLanguage, inputWord)
                    predictedWord = wordFromIndexes(outputLanguage, predictedWord)
                    groundTruth = wordFromIndexes(outputLanguage, groundTruth)
                    outCSV.loc[len(outCSV)] = [inputWord, predictedWord, groundTruth]

        if writeToCSV:
            outCSV.to_csx("predictions.csv", index=False)
                    
    accuracy = total_correct/len(dataloader)
    return accuracy, decoder_outputs


def train(train_dataloader, val_dataloader, encoder, decoder, n_epochs, learning_rate, opt):
    if opt == "adam":
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    elif opt == "rmsprop":
        encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate)
    else:
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        val_accuracy, _ = eval(encoder, decoder, val_dataloader, False)

def test(test_dataloader, encoder, decoder):
    test_accuracy, decoded_outputs = eval(encoder, decoder, test_dataloader, True)


def harness(run):
    config = run.config
    batch_size = config.batch_size
    hidden_sizes = config.encoder_decoder_layers
    cell_type = config.cell_type
    dropout = config.dropout
    attention = config.attention
    epochs = config.epochs
    inputEmbedding = config.input_enbedding_size
    outputEmbedding = config.output_embedding_size
    learning_rate = config.learning_rate
    opt = config.optimiser

    train_data_path = args.trainPath
    val_data_path = args.valPath
    test_data_path = args.testPath

    
    inputLanguage, outputLanguage, trainPairs = read_data(train_data_path, inputLanguage, outputLanguage)
    inputLanguage, outputLanguage, validPairs = read_data(val_data_path, inputLanguage, outputLanguage)
    inputLanguage, outputLanguage, testPairs = read_data(test_data_path, inputLanguage, outputLanguage)

    trainDataloader = get_dataloader(batch_size, trainPairs, inputLanguage, outputLanguage)
    valDataloader = get_dataloader(batch_size, validPairs, inputLanguage, outputLanguage)
    testDataloader = get_dataloader(batch_size, testPairs, inputLanguage, outputLanguage)

    encoder = EncoderRNN(inputLanguage.n_letters, hidden_sizes, inputEmbedding, cell_type, dropout).to(device)
    decoder = AttnDecoderRNN(outputLanguage.n_letters, hidden_sizes, outputEmbedding, cell_type, dropout, attention).to(device)

    train(trainDataloader, valDataloader, encoder, decoder, epochs, learning_rate, opt)


def train_wandb():
    with wandb.init() as run:
        harness(run)


inputLanguage = Lang("English")
outputLanguage = Lang("Hindi")
outCSV = pd.DataFrame(columns=['a', 'b', 'c'])

