import time

import torch

from language_model.evaluate.token_evaluate import TokenEvaluate

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
start_time = time.time()

lengths = list(range(2, 8)) + list(range(8, 16, 2)) + list(range(16, 32, 4)) + list(range(32, 65, 8))

samples = 10
iters = 1

offsets = list(range(0, 14))
for i in range(0, 1):
    per_iter = samples // iters
    start_position = i * per_iter
    stop_position = i * per_iter + per_iter

    # instance = TextEvaluate.TextEvaluate(device=device, samples=10000, read_lengths=lengths, candidate_lengths=lengths, offsets=offsets)
    # instance.eval(start_position, stop_position)

    # instance = TextEvaluate(device=device, samples=10, read_lengths=lengths, candidate_lengths=lengths, offsets=offsets)
    # results = instance.eval(start_position, stop_position)

    instance = TokenEvaluate(device=device, samples=10, read_lengths=[45], candidate_lengths=[5], offsets=offsets)
    results = instance.eval()

    name = '_'.join(["random_token", str(instance.vocab.vocab_size()), str(samples), "part" + str(i + 1)])
    name += ".pkl"
    results.save(name)

    del results

print('| time: {:5.2f}s |'.format(time.time() - start_time))
