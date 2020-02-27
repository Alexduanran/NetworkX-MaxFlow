import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
import copy

class MaxFlow:
    def __init__(self, input, type=None):
        if isinstance(input, int):
            randCapacity = [0 for _ in range(4)] + [i for i in range(11)]
            if type == 'complete':                                                      # Create a randomly-weighted connected graph
                nodeList = [str(i) for i in range(1, input + 1)]
                self.G = nx.complete_graph(nodeList, nx.DiGraph())
            elif type == 'dumbell':                                                     # Create a randomly-weighted dumbell graph
                randCapacity = [0 for _ in range(4)] + [i for i in range(11)]
                nodeListEven = [str(i) for i in range(1, input + 1, 2)]
                nodeListOdd = [str(i) for i in range(2, input + 1, 2)]
                self.G = nx.complete_graph(nodeListEven, nx.DiGraph())
                self.G1 = nx.complete_graph(nodeListOdd, nx.DiGraph())
                self.G.add_nodes_from(self.G1.nodes())
                self.G.add_edges_from(self.G1.edges.data())
                right = input // 2
                left = right - 1
                self.G.add_edge(str(left), str(right), weight=10)
            for x, y in self.G.edges():
                    self.G.edges[x, y]['weight'] = np.random.choice(randCapacity)
        else:                                                                           # Create a graph based on the input file
            fh=open(input, 'rb')
            self.G = nx.read_edgelist(input, create_using=nx.DiGraph)
            fh.close()

    def draw(self):
        nx.draw_networkx(self.G)

    def bfs(self, graph):
        visited = {}                                # dictionary Visited[1..n]
        for x in nx.nodes(graph):                   # set Visited[i] to false node for all nodes in graph
            visited[x] = '0'
        toExplore, S = [], []                       # List: ToExplore, S
        toExplore.append('1')                       # add source to ToExplore
        visited['1'] = '-1'                         # visited[source] = Arbitrary node
        while toExplore:                            # while ToExplore is not empty
            x = toExplore.pop(0)                    # remove node x from ToExplore
            if x == '2':                            # if sink is reached, break
                break
            for y in graph.neighbors(x):            # for each edge (x,y) in Adj(x) do
                if visited[y] == '0':               # if (Visited[y] == False)
                    visited[y] = x                  # visited[y] = previous node visited
                    toExplore.append(y)             # add y to toExplore
        sink = '2'                                  # initialize sink
        if visited[sink] == '0':                    # if sink is not reached by source
            return S                                # output empty array S
        while visited[sink] != '-1':                # while sink isn't source do
            S.insert(0, visited[sink])              # insert sink into beginning of list S
            sink = visited[sink]                    # set sink to its predesesor
        S.append('2')                               # append '2' to end of list S
        return S                                    # output S, which is the set of nodes of the path from source to sink

    def flowValue(self, path):
        weightList = []
        for i in range(len(path)-1):
            weightList.append(self.G[path[i]][path[i+1]]['weight'])             # The list of all capacity for edges in a given path
        return min(weightList)                                                  # Find the shortest capacity, which is the max flow that can pass through the path

    def ford_fulkerson_bfs(self):
        graph = self.G                                                                          # Initialize graph to be self.G
        flow = 0                                                                                # Initialize with f = 0
        while self.bfs(graph):                                                                  # while there is a flow f' in Gf with v(f') > 0
            path = self.bfs(graph)                                                              # initialize path as f'
            maxPathFlow = self.flowValue(path)                                                  # initialize maxPathFlow as v(f')
            flow += maxPathFlow                                                                 # f = f + f'
            for i in range(len(path)-1):                                                        # for every node in the path f', do
                weight = 0
                if graph.has_edge(path[i+1], path[i]):                                          # if there is a backward edge
                    weight = graph[path[i+1]][path[i]]['weight']
                    graph.edges[path[i+1], path[i]]['weight'] = maxPathFlow + weight            # the weight of backward edge gets increased by the v(f')
                else:                                                                           # if there isn't a backwrd edge
                    graph.add_edge(path[i+1], path[i], weight = maxPathFlow, color = 'green')   # add a backward edge to the residual graph with the v(f') as the capacity
                residualFlow = graph[path[i]][path[i+1]]['weight'] - maxPathFlow
                if residualFlow == 0:                                                           # if v(f') equals the capacity of the forward edge
                    graph.remove_edge(path[i], path[i+1])                                       # remove the forward edge
                else:                                                                           # else
                    graph.edges[path[i], path[i+1]]['weight'] = residualFlow                    # the capacity of the forward edge is reduced by v(f')
        return flow                                                                             # output f

    def ford_fulkerson_dijkstra(self):
        graph = self.G
        flow = 0
        try:                                                                                    # while nx.dijkstra_path(graph, '1', '2') does not throw exception NetworkXNoPath, do
            while nx.dijkstra_path(graph, '1', '2'):
                path = nx.dijkstra_path(graph, '1', '2')
                maxPathFlow = self.flowValue(path)
                flow += maxPathFlow
                for i in range(len(path)-1):
                    weight = 0
                    if graph.has_edge(path[i+1], path[i]):
                        weight = graph[path[i+1]][path[i]]['weight']
                        graph.edges[path[i+1], path[i]]['weight'] = maxPathFlow + weight
                    else:
                        graph.add_edge(path[i+1], path[i], weight = maxPathFlow, color = 'green')
                    residualFlow = graph[path[i]][path[i+1]]['weight'] - maxPathFlow
                    if residualFlow == 0:
                        graph.remove_edge(path[i], path[i+1])
                    else:
                        graph.edges[path[i], path[i+1]]['weight'] = residualFlow
        except:                                                                                 # if the error is thrown, meaning there is no longer a path from source to sink
            return flow

def benchmark(nodes, runs, type):
    nodesTimeBfs = []
    nodesTimeDijkstra = []
    nodesNum = [x for x in range(2, nodes+1)]
    for x in range(2, nodes+1):
        totalTimeBfs = 0
        totalTimeDij = 0
        for _ in range(runs):
            g = MaxFlow(x, type)
            g1 = copy.deepcopy(g)
            tbfs_start = perf_counter()
            g.ford_fulkerson_bfs()
            tbfs_stop = perf_counter()
            tdij_start = perf_counter()
            g1.ford_fulkerson_dijkstra()
            tdij_stop = perf_counter()
            totalTimeBfs += (tbfs_stop - tbfs_start)
            totalTimeDij += (tdij_stop - tdij_start)
        nodesTimeBfs.append(totalTimeBfs / runs)
        nodesTimeDijkstra.append(totalTimeDij / runs)
    plt.title('Run Time Comparison Between Ford-Fulkerson Using BFS And Dijkstra')
    plt.xlabel('number of nodes in the graph')
    plt.ylabel('run time')
    plt.plot(nodesNum, nodesTimeBfs)
    plt.plot(nodesNum, nodesTimeDijkstra)
    plt.legend(['BFS', 'Dijkstra'], loc='upper left')
    plt.show()

#----------TEST----------TEST----------TEST----------TEST----------TEST----------TEST----------TEST----------TEST----------TEST----------TEST----------TEST----------

# Initiate a class instance with a graph made from the input file 'path.txt'
#g = MaxFlow('path.txt')

# Visualize the graph
#g.draw()

# Run Ford-Fulkerson with BFS on the graph and return the max flow (One must be commented out)
#g.ford_fulkerson_bfs()

# Run Ford-Fulkerson with Dijkstra on the graph and return the max flow (One must be commented out)
#g.ford_fulkerson_dijkstra()

# Compare the performance of the two algorithms and graph them
# The first input is the total number of nodes tested
# The second input is the number of runs perform for each node
# The third input is either 'complete' or 'dumbell', which is to test the two algorithms on either a complete graph or a dumbell graph
benchmark(50, 20, 'complete')
