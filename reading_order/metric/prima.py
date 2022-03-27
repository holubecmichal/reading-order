from document.page_xml import PageXML
from reading_order.metric.prima_methods import relationship_matrix, matrix_to_pandas, fill_with_not_defined, cmp_index, \
    groups_changes, is_exception, READING_ORDER_PENALTIES, str_error
from reading_order.reading_order import ReadingOrder, get_ids


def compare(xml: PageXML, ground_truth: ReadingOrder, actual: ReadingOrder):
    gmatrix = relationship_matrix(ground_truth)
    amatrix = relationship_matrix(actual)

    dfg = matrix_to_pandas(gmatrix)
    dfa = matrix_to_pandas(amatrix)

    g_ids = set(get_ids(ground_truth.get_all_items()))
    a_ids = set(get_ids(actual.get_all_items()))

    if g_ids.difference(a_ids):
        # nejaky region je definovany v ground truth, ale neni definovany v actual
        ids = g_ids.difference(a_ids)
        dfa = fill_with_not_defined(ids, dfa)

    if a_ids.difference(g_ids):
        # nejaky region je definovany v actual, ale neni definovany v ground_truth
        ids = a_ids.difference(g_ids)
        dfg = fill_with_not_defined(ids, dfg)

    cmp = dfg.compare(dfa).rename(columns={'self': 'ground_truth', 'other': 'actual'})

    processed = []
    results = []

    for row_index, row in cmp.iterrows():
        row = row.dropna()

        for col_index in row.index.unique(0):
            cmp_i = cmp_index(row_index, col_index)

            if not groups_changes(row_index, ground_truth, actual, cmp) or not groups_changes(col_index, ground_truth,
                                                                                              actual, cmp):
                continue

            if cmp_i in processed:
                continue

            processed.append(cmp_i)
            result = cmp.loc[row_index][col_index]

            if not is_exception(result, row_index, col_index, ground_truth, actual):
                results.append((row_index, col_index, result['actual'], result['ground_truth']))

    return PrimaResults(results, xml)


class PrimaResults:
    def __init__(self, results, xml: PageXML):
        self.results = results
        self.xml = xml

    def penalty(self) -> int:
        return calculate_penalty(self.results)

    def percentage(self) -> float:
        return round(calculate_penalty_percentage(self.results, self.xml), 2)

    def errors(self) -> [str]:
        outputs = []

        for result in self.results:
            outputs.append(str_error(*result))

        return outputs

    def print(self):
        for s in self.errors():
            print(s)


def calculate_penalty(results):
    penalty = 0

    for result in results:
        _, _, actual, ground_truth = result
        penalty += READING_ORDER_PENALTIES[actual][ground_truth]

    return penalty


def calculate_penalty_percentage(results, xml: PageXML):
    max_val = 0

    for i in READING_ORDER_PENALTIES:
        row = READING_ORDER_PENALTIES[i]

        for j in row:
            max_val = max(max_val, READING_ORDER_PENALTIES[i][j])

    e50 = max_val * len(xml.get_text_regions()) / 2
    denominator = calculate_penalty(results) * (1 / e50) + 1
    return round(1 / denominator, 4) * 100


def results_to_string(results):
    outputs = []

    for result in results:
        outputs.append(str_error(*result))

    return outputs