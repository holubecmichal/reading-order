from document.stubs import Document as StubDocument
from reading_order.metric.prima_methods import relationship_matrix, matrix_to_pandas, fill_with_not_defined, cmp_index, \
    groups_changes, is_exception, READING_ORDER_PENALTIES, str_error
from reading_order.reading_order import ReadingOrder, get_ids


def compare(doc: StubDocument, ground_truth: ReadingOrder, identified: ReadingOrder) -> 'Prima':
    """
    Metrika PRIMA, funkce prijima dokument, ground truth a identifikovanou posloupnost,
    porovna dle vlastnosti metriky a vrati instanci Prima, ktera obsahuje hodnoty metriky
    """
    # ground truth matice vztahu
    gmatrix = relationship_matrix(ground_truth)
    # matice vztahu pro identifikovanou posloupnost
    amatrix = relationship_matrix(identified)

    # prevod struktury do pandas, ktera umoznuje jednoduchy diff
    dfg = matrix_to_pandas(gmatrix)
    dfa = matrix_to_pandas(amatrix)

    g_ids = set(get_ids(ground_truth.get_all_items()))
    a_ids = set(get_ids(identified.get_all_items()))

    if g_ids.difference(a_ids):
        # nejaky region je definovany v ground truth, ale neni definovany v actual, do matice se tento region
        # prida, aby bylo zohlednete pri vypoctu metriky
        ids = g_ids.difference(a_ids)
        dfa = fill_with_not_defined(ids, dfa)

    if a_ids.difference(g_ids):
        # nejaky region je definovany v actual, ale neni definovany v ground_truth, do matice se tento region
        # prida, aby bylo zohlednete pri vypoctu metriky
        ids = a_ids.difference(g_ids)
        dfg = fill_with_not_defined(ids, dfg)

    # diff
    cmp = dfg.compare(dfa).rename(columns={'self': 'ground_truth', 'other': 'actual'})

    processed = []
    results = []

    # zpracovani jednotlivych hodnot
    for row_index, row in cmp.iterrows():
        row = row.dropna()

        for col_index in row.index.unique(0):
            cmp_i = cmp_index(row_index, col_index)

            # pokud jsou prvky shodne, pokracuje se dalsim prvkem
            if not groups_changes(row_index, ground_truth, identified, cmp) or not groups_changes(col_index, ground_truth,
                                                                                                  identified, cmp):
                continue

            # pokud uz byl prvek zpracovan, neresi se a pokracuje se dalsim prvkem
            if cmp_i in processed:
                continue

            processed.append(cmp_i)
            result = cmp.loc[row_index][col_index]

            # pokud nalezeny rozdil nevyhazoval chybu v referencnim systemu Aletheia, pak ji nevyhodim ani zde
            # jinak pridavam info o lokaci a typu chyby
            if not is_exception(result, row_index, col_index, ground_truth, identified):
                results.append((row_index, col_index, result['actual'], result['ground_truth']))

    return Prima(results, doc)


class Prima:
    def __init__(self, results, doc: StubDocument):
        self.results = results
        self.doc = doc

    def penalty(self) -> int:
        # vypocet celkove hodnoty penalizace
        return calculate_penalty(self.results)

    def percentage(self) -> float:
        # procentualni hodnota chyby, hodnota metriky
        return round(calculate_penalty_percentage(self.results, self.doc), 2)

    def errors(self) -> [str]:
        # textovy vypis chyb
        outputs = []

        for result in self.results:
            outputs.append(str_error(*result))

        return outputs

    def print(self):
        for s in self.errors():
            print(s)


def calculate_penalty(results):
    """
    Funkce pro vytazeni hodnoty penalizace a vypocet celkove chyby v absolutnim cisle
    """
    penalty = 0

    for result in results:
        _, _, actual, ground_truth = result
        penalty += READING_ORDER_PENALTIES[actual][ground_truth]

    return penalty


def calculate_penalty_percentage(results, xml: StubDocument):
    """
    Funkce pro vypocet hodnoty metriky
    """
    max_val = 0

    for i in READING_ORDER_PENALTIES:
        row = READING_ORDER_PENALTIES[i]

        for j in row:
            max_val = max(max_val, READING_ORDER_PENALTIES[i][j])

    e50 = max_val * len(xml.get_text_regions()) / 2
    denominator = calculate_penalty(results) * (1 / e50) + 1
    return round(1 / denominator, 4) * 100