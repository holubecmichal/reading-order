from typing import List

from collection.Collection import Collection as OriginalCollection


class Collection(OriginalCollection):
    def group_by(self, key):
        grouped = {}
        for item in self.contents:
            grouped_by = item[key]
            if grouped_by not in grouped.keys():
                grouped[grouped_by] = [item]
            else:
                grouped[grouped_by].append(item)
        return grouped

    def pluck(self, attr: str) -> List:
        if not isinstance(self.contents, dict):
            return super(Collection, self).pluck(attr)

        return list(map(lambda x: self.contents[x][attr], self.contents))

