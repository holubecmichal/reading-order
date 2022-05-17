import time
import torch

"""
Skript pro vyhodnoceni chovani jazykoveho modelu
"""

from language_model.carrier import cs_pair_sentences, cs_vocab
from language_model.content import filter_by_shift_length, shift_pair, SHIFT_LEFT, SHIFT_RIGHT, shift_pairs
from language_model.evaluate.shift_token_evaluate import ShiftTokenEvaluate
from language_model.evaluate.text_evaluate import TextEvaluate
from language_model.evaluate.token_evaluate import TokenEvaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()

lengths = list(range(1, 8)) + list(range(8, 16, 2)) + list(range(16, 32, 4)) + list(range(32, 65, 8))

samples = 10
iters = 1

offsets = [0] + [-1] * 15
# seed pro randint - zreprodukovatelnost
torch.manual_seed(1111)

# experiment navaznosti vet
# experiment pro vyhodnoceni chovani, pokud jazykovy model odhaduje navaznost vet
# pairs = default_pair_sentences(pair_length=64, shift=5)
# left_shifted = shift_pairs(pairs, dir=SHIFT_LEFT, shift=5)
# right_shifted = shift_pairs(pairs, dir=SHIFT_RIGHT, shift=5)
# instance = ShiftTokenEvaluate(pairs=left_shifted, device=device, read_lengths=lengths, candidate_lengths=[64], offsets=offsets)

# experiment s delkou textu
# instance = TextEvaluate(device=device, samples=10, read_lengths=lengths, candidate_lengths=[64], offsets=offsets)

# experiment s delkou tokenu
instance = TokenEvaluate(device=device, samples=10000, read_lengths=lengths, candidate_lengths=[64], offsets=offsets)
instance.keep_rand_positions = True
results = instance.eval()
results.save("random_token_16_20000_10000.pkl")

print('| time: {:5.2f}s |'.format(time.time() - start_time))
