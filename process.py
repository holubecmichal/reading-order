import argparse
import os
import sys

from document import page_xml
from language_model.analyzer import LmAnalyzer
from language_model.carrier import cs_model, cs_vocab, de_model, de_vocab
from reading_order.metric.recall import compare as dp_compare
from reading_order.metric.prima import compare as prima_compare
from spatial.analyzer import DiagonalAnalyzer, ColumnarAnalyzer, ColumnarLmAnalyzer, \
    TopToBottomAnalyzer
from spatial.plotter import Plotter

ALLOWED_METHODS = ['LH', 'LS', 'SD', 'SC', 'CH', 'CS', 'TB']
TOKENS_REQUIRED_METHODS = ['L', 'C']
ALLOWED_MODELS = ['cs', 'de']

parser = argparse.ArgumentParser(description="""\
Script for analyse PageXML file to identify reading order of document.
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('path', type=str, help='Path to file which will be analysed')
parser.add_argument('method', type=str, help="""\
Method for analyse
LH - language analyse with language model, hard token decision
LS - language analyse with language model, score
SD - spatial diagonal analyse
SC - spatial columnar analyse
CH - combined analyse, hard token decision
CS - combined analyse, score
TB - top-to-bottom analyse
""")
parser.add_argument('--ground-truth', '-g', type=str,
                    help='Path to file with ground truth, metrics Prima and Recall will be show')
parser.add_argument('--tokens', '-t', type=int, help='Number of tokens, using for L and C analyse, default=3', default=3)
parser.add_argument('--model', '-m', default='cs', type=str, help="""\
Model used by language analyse or combined analyze
cs - 'Czech language model'
de - 'German language model'
""")

args = parser.parse_args()
path = os.path.abspath(args.path)

if not os.path.isfile(path):
    print('File not found {}'.format(args.path), file=sys.stderr)
    exit(1)

doc = page_xml.parse(path)

if args.method not in ALLOWED_METHODS:
    print('Unknown method {}'.format(args.method), file=sys.stderr)
    exit(1)

if args.model not in ALLOWED_MODELS:
    print('Unknown model {}'.format(args.method), file=sys.stderr)
    exit(1)

if args.method[1] in ['H', 'S']:
    if args.model == 'cs':
        model, vocab = cs_model(), cs_vocab()
    else:
        model, vocab = de_model(), de_vocab()

    lm_analyzer = LmAnalyzer(model, vocab)

    if args.method[1] == 'H':
        lm_analyzer.use_hard_limit(args.tokens)
    else:
        lm_analyzer.use_score_hard_limit(args.tokens)

    if args.method[0] == 'L':
        ro = lm_analyzer.analyze(doc)
    elif args.method[0] == 'C':
        analyzer = ColumnarLmAnalyzer()
        ro = analyzer.analyze(doc, lm_analyzer)
elif args.method == 'SD':
    analyzer = DiagonalAnalyzer()
    ro = analyzer.analyze(doc)
elif args.method == 'SC':
    analyzer = ColumnarAnalyzer()
    ro = analyzer.analyze(doc)
elif args.method == 'TB':
    analyzer = TopToBottomAnalyzer()
    ro = analyzer.analyze(doc)
else:
    print('Method not recognized', file=sys.stderr)
    exit(1)

plotter = Plotter(doc)
plotter.plot_document_border()
plotter.plot_all_regions()
plotter.annotate_text_regions()
plotter.plot_reading_order(ro)
plotter.remove_axis()

plotter.show()

if args.ground_truth:
    path = os.path.abspath(args.ground_truth)

    if not os.path.isfile(path):
        print('Ground truth not found {}'.format(path), file=sys.stderr)
        exit(1)

    gt = page_xml.parse(path)

    if not gt.get_reading_order():
        print('Ground truth is not in file {}'.format(path), file=sys.stderr)
        exit(1)

    prima_results = prima_compare(gt, gt.get_reading_order(), ro)
    print('prima penalty: {}'.format(prima_results.penalty()))
    print('prima penalty percentage: {}%'.format(prima_results.percentage()))

    dp_results = dp_compare(gt.get_reading_order(), ro)
    print('recall: {}%'.format(dp_results.recall()))
