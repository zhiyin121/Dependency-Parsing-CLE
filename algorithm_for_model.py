import numpy as np
import copy

from numpy.core.records import recarray


def get_edge(vertexs):
    edges = set()
    for i in vertexs:
        for j in vertexs:
            if i != j:
                if j != 0:
                    edges.add((i, j))
    return edges


def get_vertexs(edges):
    vertexs_set = set()
    for edge in edges:
        vertexs_set.add(edge[0])
        vertexs_set.add(edge[1])
    return vertexs_set


def cle(edges_set, cycle_list, cycle_edges_list, score_matrix):
    # find the best edges
    best_edges = set()
    vertexs = get_vertexs(edges_set)
    new_score_best_edges = np.ones(score_matrix.shape) * -np.inf
    for edge in edges_set: 
        new_score_best_edges[edge[1],edge[0]] = score_matrix[edge[1],edge[0]]
    best_edges.add((0,np.argmax(new_score_best_edges[:,0])))
    for i, best in enumerate(np.argmax(new_score_best_edges, axis = 1)):
        if i != 0 and (new_score_best_edges[i] > -np.inf).any() and i not in [x[1] for x in best_edges]:
            best_edges.add((best, i))

    if findonecycle(best_edges) == None:
        return best_edges
    else:
        cycle_list.append(findonecycle(best_edges)[0])
        cycle_edges_list.append(findonecycle(best_edges)[1])
        print('cycle_list', cycle_list)
        
        g_c, new_score = contract(edges_set, cycle_list[-1], cycle_edges_list[-1], score_matrix)        
        outside_cycle_set = cle(g_c, cycle_list, cycle_edges_list, new_score)
        resolve = resolvecycle(outside_cycle_set, cycle_list, cycle_edges_list)
        return resolve


def chu_liu_edmonds(score_matrix):
    cycle_list = []
    cycle_edges_list = []
    token_number = score_matrix.shape[1]
    # find all edges
    token_list = set(range(token_number))
    all_edges = get_edge(token_list)
    for i in token_list:
        score_matrix[i][i] = -np.inf
        
    result = cle(all_edges, cycle_list, cycle_edges_list, score_matrix)
    
    # take second element for sort
    def takeSecond(elem):
        return elem[1]
    result = list(result)
    result.sort(key=takeSecond)
    print('Final return edges:\n', result)
    return [0] + list([x[0] for x in result])


def findonecycle(edges_set):
    def get_next(edges, child):
        next_edge = []
        for edge in edges:
            if child == edge[0]:
                next_edge = edge
        if next_edge in lst:
            pass
        else:
            if next_edge != []:
                if next_edge[1] != lst[0][0]:
                    lst.append(next_edge)
                    get_next(edges, next_edge[1])
                else:
                    lst.append(next_edge)
        return lst


    for edge in edges_set:
        lst = [edge]
        get_next(edges_set, edge[1])
        if lst[0][0] == lst[-1][1]:
            v_c = set()
            for e in lst:
                v_c.add(e[0])
                v_c.add(e[1])
            return v_c, lst   


def contract(edges_set, cycle_set, cycle_edges_set, score_matrix):
    all_v = set((range(score_matrix.shape[0])))
    
    # get vertexs out of cycle
    v_d = all_v - cycle_set
    
    # combine the notes of cycle into a new note
    new_node = tuple(cycle_set)
    
    # add the new note, get edges except those in cycle
    v_d.add(new_node)
    edges = get_edge(v_d)

    g_c = set()
    # convert cycle edges to a dictionary
    dic_cycle = {}
    for edge in cycle_edges_set:
        dic_cycle[edge[1]] = edge[0]  # key:child, value:head

    w_c = 0
    for edge in cycle_edges_set:
        w_c += score_matrix[edge[1], edge[0]]

    new_score = np.ones(score_matrix.shape) * -np.inf # copy.deepcopy(score_matrix)
    d_edges_c = {}
    for edge in edges:
        if type(edge[0]) is tuple:
            c_edges_d = {}
            for head in edge[0]:
                c_edges_d[(head, edge[1])] = score_matrix[edge[1]][head]
            highest_head = max(c_edges_d, key=c_edges_d.get)
            g_c.add(highest_head)
            new_score[highest_head[1], highest_head[0]] = score_matrix[highest_head[1], highest_head[0]]
            
        elif type(edge[1]) is tuple:
            for child in edge[1]:
                d_edges_c[(edge[0], child)] = score_matrix[child][edge[0]] - score_matrix[child][dic_cycle[child]] + w_c
        else:
            g_c.add(edge)
            new_score[edge[1], edge[0]] = score_matrix[edge[1], edge[0]]
            
    highest_child = max(d_edges_c, key=d_edges_c.get)
    g_c.add(highest_child)
    new_score[highest_child[1], highest_child[0]] = d_edges_c[highest_child]

    return g_c, new_score


def resolvecycle(outside_cycle_set, cycle_list, cycle_edges_list):
    for i in list(range(-1, -len(cycle_list)-1, -1)):# range(len(cycle_list)-1,-1,-1):
    # for cycle_set, cycle_edges_set in zip(reversed(cycle_list), reversed(cycle_edges_list)):
        cycle_set = cycle_list[i]
        cycle_edges_set = cycle_edges_list[i]
        if findonecycle(cycle_edges_set) == None: break

        # convert cycle edges to a dictionary
        dic_cycle = {}
        for edge in cycle_edges_set:
            dic_cycle[edge[1]] = edge[0]  # key:child, value:head
        
        for edge in outside_cycle_set:
            if edge[1] in cycle_set:
                cycle_edges_set.remove((dic_cycle[edge[1]], edge[1]))
                break
        outside_cycle_set = outside_cycle_set.union(cycle_edges_set)

    return outside_cycle_set


#------------------test--------------------

score_matrix = np.array([[0, 0, 0, 0],[9, 0, 30, 11], [10, 20, 0, 0], [9, 3, 30, 0]], dtype=float)
#score_matrix = np.array([[0, 0, 0, 0, 0], [9, 0, 11, 30, 2], [10, 20, 0, 0, 5], [9, 3, 30, 0, 2], [2, 4, 5, 7, 8]], dtype=float)
#score_matrix = np.random.random((7,7))

heads = chu_liu_edmonds(score_matrix)
print('Head list(including \'ROOT\'):\n', heads)


