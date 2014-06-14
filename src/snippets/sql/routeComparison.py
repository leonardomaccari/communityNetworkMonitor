
from collections import defaultdict
import networkx as nx
from mpr import *

def getRoutingTable(node, G, metric="weight", cutoff=1):
    shortestPaths = nx.single_source_dijkstra(G, node, weight=metric)[1]
    rTable = {}
    for dest, path in shortestPaths.items():
        if len(path) > cutoff:
            rTable[dest] = path[1]
    return rTable
    
def navigateGraph(rTable,  source, dest, sol = []):
    """ This graph computes the route that a packet would really perform.
    It navigates the routing tables of each node and finds its way to 
    destination. 
    TODO: it is a recursive function and it passes the whole routing table
    as a parameter. this is not memory-efficient"""

    if source == dest:
        return sol
    else:
        nh = rTable[source][dest]
        sol.append(nh)
        return navigateGraph(rTable, nh, dest, sol)
 
def routeComparison(G, solution, metric = "weight"):
    """ Compare the best route available with the route chosen by the 
    routing protocol """

    weightStats = defaultdict(list)
    allp = nx.all_pairs_dijkstra_path_length(G, weight=metric)
    rTable = {}
    for node in G.nodes():
        rTable[node] = getRoutingTable(node,
               getNodeView(node, solution,  G, metric=metric))
    for i in range(len(G)):
        source = G.nodes()[i]
        for k in range(i+1, len(G)):
            target = G.nodes()[k]
            sol = navigateGraph(rTable, source, target, [source])
            weight = 0
            for k in range(len(sol)-1):
                w = G[sol[k]][sol[k+1]][metric]
                weight += w
            weightStats[(source,target)].append(
                    [allp[source][target],weight])
    return weightStats
       

