import argparse
import os.path
import pickle

from document.page_xml import parse
from language_model.carrier import default_model, default_vocab, de_model, de_vocab
from reading_order.processor import Processor
from spatial.spare import SpaRe

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str, required=True)

args = parser.parse_args()
path = os.path.abspath(args.path)

processor = Processor(de_model(), de_vocab())
errors = []


def save(dir, file, content: dict):
    filename = file.replace('.xml', '') + '.pkl'
    with open(os.path.join(dir, filename), 'wb') as f:
        pickle.dump(content, f)


if os.path.isfile(path):
    files = [os.path.basename(path)]
    dir = os.path.dirname(path)
else:
    dir = path
    files = []

    for filename in os.listdir(path):
        if not filename.endswith('.xml'):
            continue

        files.append(filename)

if not files:
    print('Not files specified')
    exit()

for i, file in enumerate(sorted(files)):
    print("{}: {}".format(i, file))

    try:
        filepath = os.path.join(dir, file)
        xml = parse(filepath)

        spare = SpaRe(xml)
        results = {
            'top-to-bottom': spare.get_top_to_bottom_reading_order(),
            'ocr': spare.get_ocr_reading_order(),
            'diagonal': spare.get_diagonal_reading_order(),
            'columnar': spare.get_columnar_reading_order(),
            'columnar-heading': spare.get_columnar_heading_reading_order(),
            'columnar-lm': spare.get_columnar_lm_reading_order(processor)
        }

        # print('lm-2')
        # results['lm-2'] = processor.process(xml.get_text_regions(), token_limit=2)
        # processor.clear()
        # print('lm-10')
        # results['lm-10'] = processor.process(xml.get_text_regions(), token_limit=10)
        # processor.clear()

        save(dir, file, results)
    except Exception as e:
        s = str(e)
        err = "{} - {}".format(file, s)
        errors.append(err)
        print(err)


for err in errors:
    print(err)
