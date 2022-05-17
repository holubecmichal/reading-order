# coding: utf-8
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.onnx

import model
from language_model.content_tokens import load_content_tokens
from language_model.constants import get_model_path
from language_model.vocabulary import load_vocab

# --emsize=400 --nhid=1700 --cuda=1 --vocab_size=20000

"""
Skript pro trenovani jazykoveho modelu
Prevzato z https://github.com/pytorch/examples/tree/main/word_language_model a upraveno pro ucely prace
"""

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--root', type=str, default='./cs')

parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=int,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--continue_training', default=False, action='store_true',
                    help='continue training when model exists')
parser.add_argument('--vocab_size', default=8000, type=int,
                    help='continue training when model exists')

args = parser.parse_args()

model_name_parameters = [
    args.model,
    'vocabsize' + str(args.vocab_size),
    'emsize' + str(args.emsize),
    'nhid' + str(args.nhid),
    'nlayers' + str(args.nlayers),
    'dropout' + str(args.dropout),
    'batchsize' + str(args.batch_size),
    'seqlen' + str(args.bptt)
]

model_name = '_'.join(model_name_parameters) + '.tar'
root = os.path.abspath(args.root)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not isinstance(args.cuda, int):
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:" + str(args.cuda) if isinstance(args.cuda, int) else "cpu")

###############################################################################
# Build the model
###############################################################################

ntokens = args.vocab_size
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

print(model)
print(model_name)

if args.continue_training and os.path.isfile(get_model_path(root, model_name)):
    checkpoint = torch.load(get_model_path(root, model_name), map_location=torch.device(device))
    best_val_loss = checkpoint['loss']
    last_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    acc = checkpoint['acc']

    print('=' * 93)
    print('| continue training | last epoch {:3d} | loss {:5.4f} | valid ppl {:8.4f} | accurancy {:5.4f}'
          .format(checkpoint['epoch'], best_val_loss, math.exp(best_val_loss), acc * 100))
    print('=' * 93)
else:
    best_val_loss = None
    last_epoch = 1

criterion = nn.NLLLoss()

###############################################################################
# Load data
###############################################################################

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

eval_batch_size = 10
vocab = load_vocab(root, 'train.txt', args.vocab_size)
print('load train')
train_data = load_content_tokens(root, 'train.txt', vocab, device).batchify(args.batch_size)
print('load validate')
val_data = load_content_tokens(root, 'valid.txt', vocab, device).batchify(eval_batch_size)
print('load test')
test_data = load_content_tokens(root, 'test.txt', vocab, device).batchify(eval_batch_size)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data.to(device), target.to(device)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        x = 0
        acc = 0

        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()

            acc += ((targets == output.argmax(dim=1)).sum()) / targets.shape[0]
            x += 1

    return total_loss / (len(data_source) - 1), acc / x

batches = len(train_data) // args.bptt
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()

    hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time

            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                  'loss {:5.4f} | ppl {:8.4f} '.format(epoch, batch, batches, lr, elapsed * 1000 / args.log_interval,
                                                       cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


# Loop over epochs.
lr = args.lr

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(last_epoch, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss, acc = evaluate(val_data)
        print('-' * 93)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '
                'valid ppl {:8.4f} | accurancy {:5.4f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss), acc * 100))
        print('-' * 93)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(get_model_path(root, model_name), 'wb') as f:
                # model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
                #                        args.tied).to(device)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': val_loss,
                    'acc': acc,
                    'model': args.model,
                    'ntoken': ntokens,
                    'emsize': args.emsize,
                    'nhid': args.nhid,
                    'nlayers': args.nlayers,
                    'dropout': args.dropout,
                    'tied': args.tied
                }, f)

            best_val_loss = val_loss
        # else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            # lr /= 2
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(model_name, 'rb') as f:
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss, acc = evaluate(test_data)
print('=' * 93)
print('| End of training | test loss {:5.4f} | test ppl {:8.4f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 93)
