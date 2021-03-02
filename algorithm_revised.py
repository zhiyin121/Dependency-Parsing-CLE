import numpy as np

def get_edge(vertexs):
        edges = set()
        for i in vertexs:
            for j in vertexs:
                if i != j and j != 0:
                    edges.add((i, j))
        return edges


class Graph(object):
    def __init__(self, vertexs, edges):
        self.vertexs = vertexs
        self.edges =  edges

        
def embedding(vertex):
    embed_dic = {0: [0, 1, 0], 1: [1, 2, 3], 2: [2, 4, 2], 3: [0, 2, 1]}
    return embed_dic[vertex]


def sigma(edges):
    scores = []
    for edge in edges:
        embed_head = embedding(edge[0])
        embed_depentdent = embedding(edge[1])
        score = round(cosine_similarity(np.array(embed_head), np.array(embed_depentdent)), 2)
        scores.append(score)
    return scores
            

def chuliuedmonds(g, sigma):
    a = set()
    dic = {}
    for v_d in g.vertexs:
        for edge in g.edges:
            if v_d == edge[1] and v_d != 0:
                dic[edge[0]] = sigma([edge])
        if dic != {}:
            v_h = max(dic, key=dic.get)
            a.add((v_h, v_d))
        dic = {}
    g_a = Graph(g.vertexs, a)
    if findonecycle(g_a.edges) == None:
        print(g_a.edges)
        return g_a.edges
    else:
        c = set(findonecycle(g_a.edges))
        g_c = contract(g, c, sigma)
        y = chuliuedmonds(g_c, sigma)
        
        print(resolvecycle(y, c))
        return resolvecycle(y, c)
        

def findonecycle(g_a):
        
    def get_next(g_a, dependent):
        next_pair = []
        for i in g_a:
            if dependent == i[0]:
                next_pair = i
        if next_pair != []:
            if next_pair[1] != lst[0][0]:
                lst.append(next_pair)
                get_next(g_a, next_pair[1])
            else:
                lst.append(next_pair)
        return lst

    for i in g_a:
        lst = [i]
        get_next(g_a, i[1])
        if lst[0][0] == lst[-1][1]:
            return lst


def contract(g, c, sigma):
    v_c = set()
    dic_c = {}
    for i in c:
        v_c.add(i[0])
        v_c.add(i[1])
        dic_c[i[1]] = i[0] #key:dependent, value:head
     
    all_v_d = g.vertexs - v_c
    g_c = set()
    for i in g.edges:
        if i[0] in v_c or i[1] in v_c:
            pass
        else:
            g_c.add(i)
    
    for v_d in all_v_d:
        if v_d == 0:
            pass
        else:
            have_head = []
            for j in g_c:
                if v_d == j[1]:
                   have_head.append(v_d)
            if have_head != None:
                pass
            else:
                dic_h = {}
                for v_h in v_c:
                    h_score = sigma([[v_h, v_d]])[0]
                    dic_h[v_h] = h_score
                v_h = max(dic_h, key=dic_h.get)
                g_c.add((v_h, v_d))

    for v_h in all_v_d:
        root_child = []
        for j in g_c:
            if j[0] == 0:
                root_child.append(j)
                
        if v_h == 0 and root_child != None:
            pass
        else:
            dic_d = {}
            for v_d in v_c:
                d_score = sigma([[v_h, v_d]])[0] - sigma([[dic_c[v_d], v_d]])[0]
                dic_d[v_d] = d_score
            v_d = max(dic_d, key=dic_d.get)
            g_c.add((v_h, v_d))

    return Graph(g.vertexs, g_c)

    
def resolvecycle(y, c):
    v_c = set()
    dic_c = {}
    for i in c:
        v_c.add(i[0])
        v_c.add(i[1])
        dic_c[i[1]] = i[0] #key:dependent, value:head

    result = y
    for e_y in y:
        v_d = e_y[1]
        if v_d in v_c:
            del_edge = dic_c[v_d], v_d
    for e_c in c:
        if e_c != del_edge:
            result.add(e_c)
        else:
            pass
    return result

    
    

#--------------test-----------------
    
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

vertexs = {0, 1, 2, 3}
edges = get_edge(vertexs)
g = Graph(vertexs, edges)
chuliuedmonds(g, sigma)
