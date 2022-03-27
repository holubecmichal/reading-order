import torch

from e_results import EResults
from language_model.evaluate.results_graphs import ResultsGraphs

path = 'cs/results/'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# path_chunks = path + 'word/'
path_chunks = path + 'token/'
# path_chunks = path + 'random_token/'
results = []

# i = 1
# for filename in os.listdir(path_chunks):
#     if not filename.endswith(".pkl"):
#         continue
#
#     print(str(i) + ': ' + filename)
#     i += 1
#     eresults = EResults.load(path_chunks, filename, device)
#     results += eresults.results
#
# eresults.results = results

# eresults = EResults.load(path, "token_20000_1000.pkl", device)
# eresults = EResults.load(path, "wideoffset_20000_10000.pkl", device)

# eresults = EResults.load(path, "20000_50.pkl", device)
# eresults = EResults.load(path, "20000_100.pkl", device)
eresults = EResults.load(path, "20000_10.pkl")
# eresults = EResults.load(path, "random_token/random_token_20000_10000_part1.pkl", device)
# eresults = EResults.load(path, "random_20000_10000.pkl", device)
# eresults.print_result()

print('processing ...')

collection = eresults.get_results()
resultsGraphs = ResultsGraphs(collection)

title = 'temp title'
resultsGraphs.heatmap(title)
resultsGraphs.tendency(title)
resultsGraphs.total_result_bar(title)
resultsGraphs.bar_area(title, [1, 2, 64], [1, 2, 64])

# tokens = {}
# with open('./cs/wiki-czech/train.txt', 'r', encoding="utf8") as f:
#     for line in f:
#         words = line.split()
#         for word in words:
#             if word not in tokens:
#                 tokens[word] = eresults.cs.corpus.sp.Encode(word)
#
# count = {}
# nump = []
# for word, tokenized in tokens.items():
#     length = len(tokenized)
#
#     if length not in count:
#         count[length] = 0
#
#     count[length] += 1
#     nump.append(length)
#
# nump = np.array(nump)
# count_result = {}
# for i in range(1, 11):
#     count_result[i] = count[i]
#
#
# plt.bar(count_result.keys(), count_result.values())
# plt.xticks(list(count_result.keys()))
# plt.ylabel('Četnost')
# plt.xlabel('Počet tokenů na slovo')
#
# for i, val in count_result.items():
#     if i == 9:
#         continue
#
#     plt.text(i, val + 10000, format(val, ",").replace(',', ' '), ha = 'center')
#
# # plt.grid(axis='y')
# # plt.show()
# # plt.savefig('train_split_frequency.eps', format='eps')