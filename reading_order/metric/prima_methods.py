from pandas import DataFrame

from reading_order.reading_order import ReadingOrder, Item, flatten_items, get_nearest_ancestor, get_group_which_has_item
import numpy as np
import pandas as pd


class UnknownRelationship(Exception):
    ...


REL_SAME = '='
REL_SUCCESSOR = '->'
REL_PREDECESSOR = '<-'
REL_NEITHER_DIRECT_NOR_UNORDERED = '-x-'
REL_SOMEWHERE_BEFORE = '->->'
REL_SOMEWHERE_AFTER = '<-<-'
REL_FULLY_UNORDERED = '--'
REL_NOT_DEFINED = 'n.d.'


READING_ORDER_PENALTIES = {
    REL_SUCCESSOR:                    {REL_SUCCESSOR: 0,  REL_PREDECESSOR: 40, REL_FULLY_UNORDERED: 10, REL_NEITHER_DIRECT_NOR_UNORDERED: 20, REL_NOT_DEFINED: 0, REL_SOMEWHERE_BEFORE: 0,  REL_SOMEWHERE_AFTER: 10},
    REL_PREDECESSOR:                  {REL_SUCCESSOR: 40, REL_PREDECESSOR: 0,  REL_FULLY_UNORDERED: 10, REL_NEITHER_DIRECT_NOR_UNORDERED: 20, REL_NOT_DEFINED: 0, REL_SOMEWHERE_BEFORE: 10, REL_SOMEWHERE_AFTER: 0},
    REL_FULLY_UNORDERED:              {REL_SUCCESSOR: 20, REL_PREDECESSOR: 20, REL_FULLY_UNORDERED: 0,  REL_NEITHER_DIRECT_NOR_UNORDERED: 10, REL_NOT_DEFINED: 0, REL_SOMEWHERE_BEFORE: 10, REL_SOMEWHERE_AFTER: 10},
    REL_NEITHER_DIRECT_NOR_UNORDERED: {REL_SUCCESSOR: 20, REL_PREDECESSOR: 20, REL_FULLY_UNORDERED: 10, REL_NEITHER_DIRECT_NOR_UNORDERED: 0,  REL_NOT_DEFINED: 0, REL_SOMEWHERE_BEFORE: 10, REL_SOMEWHERE_AFTER: 10},
    REL_NOT_DEFINED:                  {REL_SUCCESSOR: 20, REL_PREDECESSOR: 20, REL_FULLY_UNORDERED: 10, REL_NEITHER_DIRECT_NOR_UNORDERED: 0,  REL_NOT_DEFINED: 0, REL_SOMEWHERE_BEFORE: 10, REL_SOMEWHERE_AFTER: 10},
    REL_SOMEWHERE_BEFORE:             {REL_SUCCESSOR: 0,  REL_PREDECESSOR: 20, REL_FULLY_UNORDERED: 5,  REL_NEITHER_DIRECT_NOR_UNORDERED: 5,  REL_NOT_DEFINED: 0, REL_SOMEWHERE_BEFORE: 0,  REL_SOMEWHERE_AFTER: 10},
    REL_SOMEWHERE_AFTER:              {REL_SUCCESSOR: 20, REL_PREDECESSOR: 0,  REL_FULLY_UNORDERED: 5,  REL_NEITHER_DIRECT_NOR_UNORDERED: 5,  REL_NOT_DEFINED: 0, REL_SOMEWHERE_BEFORE: 10, REL_SOMEWHERE_AFTER: 0},
}


def check_relationship(source: Item, checking: Item) -> str:
    if source == checking:
        return REL_SAME

    if source.get_successor() and source.get_successor() == checking:
        return REL_SUCCESSOR

    if source.get_predecessor() and source.get_predecessor() == checking:
        return REL_PREDECESSOR

    if source.get_parent() == checking.get_parent():
        if source.get_parent().is_ordered():
            return REL_NEITHER_DIRECT_NOR_UNORDERED
        elif source.get_parent().is_unordered():
            return REL_FULLY_UNORDERED

    ancestor = get_nearest_ancestor(source, checking)
    if ancestor.is_unordered():
        if source.get_parent().is_ordered() and checking.get_parent().is_ordered():
            if source.get_predecessor() is None and checking.get_successor() is None:
                return REL_SOMEWHERE_AFTER
            elif source.get_successor() is None and checking.get_predecessor() is None:
                return REL_SOMEWHERE_BEFORE
            else:
                return REL_NEITHER_DIRECT_NOR_UNORDERED

        return REL_FULLY_UNORDERED

    if ancestor.is_ordered():
        groups = list(ancestor.get_only_groups().values())

        if source.get_level() < checking.get_level():
            checking = get_group_which_has_item(groups, checking.get_id())

            if checking.is_unordered():
                return REL_SOMEWHERE_BEFORE

            return check_relationship(source, checking)
        elif source.get_level() > checking.get_level():
            source = get_group_which_has_item(groups, source.get_id())

            if source.is_unordered():
                return REL_SOMEWHERE_AFTER

            return check_relationship(source, checking)
        else:
            return REL_NEITHER_DIRECT_NOR_UNORDERED

    raise UnknownRelationship(source.get_id() + ' - ' + checking.get_id())


def relationship_matrix(reading_order: ReadingOrder):
    items = reading_order.get_all_items()
    items.sort(key=lambda item: item.get_id())
    matrix = []

    for source in items:
        row = [source.get_id()]

        for checking in items:
            relation = check_relationship(source, checking)
            row.append(relation)

        matrix.append(row)

    return matrix


def matrix_to_pandas(matrix):
    matrix = np.array(matrix)
    ids = matrix[:, 0]

    return pd.DataFrame(matrix[:, 1:], index=ids, columns=ids)


def str_error(lid, rid, arelation, grelation):
    return 'Ground Truth region {} - region {}: Relation [{}] instead of [{}]'.format(lid,
                                                                                      rid,
                                                                                      arelation,
                                                                                      grelation)


def cmp_index(index1, index2):
    lst = [index1, index2]
    lst.sort()
    return '-'.join(lst)


def is_exception(result, row_index, col_index, ground_truth: ReadingOrder, actual: ReadingOrder):
    if result['ground_truth'] == REL_SUCCESSOR and result['actual'] == REL_FULLY_UNORDERED:
        gleft = ground_truth.get_by_id(row_index)
        gright = ground_truth.get_by_id(col_index)
        aleft = actual.get_by_id(row_index)
        aright = actual.get_by_id(col_index)

        if len({gleft.get_level(), gright.get_level(), aleft.get_level(), aright.get_level()}) == 1:
            # vsichni jsou na stejne urovni

            if gleft.get_parent().get_id() == gright.get_parent().get_id():
                if gleft.get_parent().is_ordered() and aleft.get_parent().is_ordered():
                    if gleft.get_predecessor() is None:
                        return True

                    if gright.get_successor() is None:
                        return True
        else:
            gitems = flatten_items(gleft.get_parent())
            if len(gitems) != 2 and aleft.get_level() == 1 and gleft.get_parent().is_ordered() and aleft.get_parent().is_unordered():
                return True

            if len(gitems) != 2 and aright.get_level() == 1 and gleft.get_parent().is_ordered() and aright.get_parent().is_unordered():
                return True

    if result['ground_truth'] == REL_NEITHER_DIRECT_NOR_UNORDERED and result['actual'] == REL_FULLY_UNORDERED:
        # --
        # vytazeny puvodne serazeny prvek do nadrazeneho neserazene skupiny
        gleft = ground_truth.get_by_id(row_index)
        gright = ground_truth.get_by_id(col_index)
        aleft = actual.get_by_id(row_index)
        aright = actual.get_by_id(col_index)

        if gleft.get_parent().is_ordered() and aleft.get_parent().is_unordered() and gleft.get_level() > aleft.get_level():
            return True

        if gright.get_parent().is_ordered() and aright.get_parent().is_unordered() and gright.get_level() > aright.get_level():
            return True

    if result['ground_truth'] == REL_NEITHER_DIRECT_NOR_UNORDERED and result['actual'] == REL_SOMEWHERE_BEFORE:
        # ->->
        # vnorena unordered skupina, ve ktere jsou polozky, ktere byly drive na stejne urovni a ordered
        return True

    if result['ground_truth'] == REL_SUCCESSOR and result['actual'] == REL_SOMEWHERE_BEFORE:
        # puvodne serazeny prvek je nove v neserazene skupine, ktera nahradila jeho misto v serazene skupine - creepy popis
        return True

    if (result['ground_truth'] == REL_FULLY_UNORDERED and result['actual'] == REL_NEITHER_DIRECT_NOR_UNORDERED) \
            or (result['ground_truth'] == REL_FULLY_UNORDERED and result['actual'] == REL_SUCCESSOR):
        # puvodne prvni prvek serazene skupiny prehozen

        gleft = ground_truth.get_by_id(row_index)
        aleft = actual.get_by_id(row_index)

        gright = ground_truth.get_by_id(col_index)
        aright = actual.get_by_id(col_index)

        if len({gleft.get_level(), gright.get_level(), aleft.get_level(), aright.get_level()}) == 1:
            if gright.get_parent().get_id() != aright.get_parent().get_id():
                if gright.get_parent().is_ordered() and aright.get_parent().is_ordered():
                    if gright.get_predecessor() is None and aright.get_successor() is None:
                        return True

    if result['ground_truth'] == REL_SOMEWHERE_AFTER or result['ground_truth'] == REL_SOMEWHERE_BEFORE:
        if result['actual'] == REL_NEITHER_DIRECT_NOR_UNORDERED:
            return True

        if result['actual'] == REL_FULLY_UNORDERED:
            return True

        if result['actual'] == REL_SUCCESSOR:
            return True

    if result['actual'] == REL_SOMEWHERE_AFTER or result['actual'] == REL_SOMEWHERE_BEFORE:
        if result['ground_truth'] == REL_NEITHER_DIRECT_NOR_UNORDERED:
            return True


def fill_with_not_defined(ids, df) -> DataFrame:
    for id in ids:
        df[id] = REL_NOT_DEFINED
        df.loc[id] = REL_NOT_DEFINED
        df.loc[id][id] = REL_SAME

    df = df.reindex(sorted(df.columns), axis=0)
    df = df.reindex(sorted(df.columns), axis=1)

    return df


def groups_changes(id, ground_truth: ReadingOrder, actual: ReadingOrder, cmp) -> bool:
    gitem = ground_truth.get_by_id(id)

    if gitem is None:
        return True

    aitem = actual.get_by_id(id)

    if aitem is None:
        return True

    group = gitem.get_parent()

    if len(group.items) != len(actual.get_by_id(id).get_parent().items):
        return True

    for i in group.items:
        a = group.items[i]

        for j in group.items:
            b = group.items[j]

            try:
                val = cmp.loc[a.get_id()][b.get_id()]

                if val.dropna().empty is False:
                    return True
            except KeyError:
                pass

    return False
