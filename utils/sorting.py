from collections import defaultdict

def topological_sort(before_in_reading: [tuple]):
    grp = Graph(len(before_in_reading))
    for t in before_in_reading:
        grp.addEdge(t[1], t[0])

    return grp.topologicalSort()

# source https://pythonwife.com/topological-sort-algorithm-in-python/
class Graph:
    def __init__(self, numberofVertices):
        self.graph = defaultdict(list)
        self.numberofVertices = numberofVertices

    def addEdge(self, vertex, edge):
        self.graph[vertex].append(edge)


    def topogologicalSortUtil(self, v, visited, stack):
        visited.append(v)

        for i in self.graph[v]:
            if i not in visited:
                self.topogologicalSortUtil(i, visited, stack)

        stack.insert(0, v)


    def topologicalSort(self):
        visited = []
        stack = []

        for k in list(self.graph):
            if k not in visited:
                self.topogologicalSortUtil(k, visited, stack)

        return list(reversed(stack))