import numpy as np
from tools import reader
from feature_revised import FeatureMapping

def get_edge(vertexs):
    edges = set()
    for i in vertexs:
        for j in vertexs:
            if i != j and j.form != 'ROOT':
                edges.add((i, j))
    return edges



class Graph(object):
    def __init__(self, vertexs, edges):  # !!!V:<v_h, v_d>, E: representation of edges
        self.vertexs = vertexs
        self.edges =  edges


def sigma(vertexs, fm):
    dic_e = fm.get_dict(vertexs)  # how to transport feature dict???
    return dic_e


def chuliuedmonds(w, g, sigma, fm):
    
    represetation = sigma(g.vertexs, fm)  # a dict {(head, dependent):[1, 23,... 45],...}

    a = set()
    dic = {}; dic_root = {}

    for v_d in g.vertexs:
        for edge in g.edges:
            if v_d.form == 'ROOT':
                r_score = dot(w, represetation[edge])
                dic_root[edge[1]] = r_score
                root_h = v_d
            else:
                if edge[1] == v_d and edge[0] != 'ROOT':
                    dic[edge[0]] = dot(w, represetation[edge])
                    # print(dic[edge[0]])
                    
            if dic != {}:
                v_h = max(dic, key=dic.get)
                a.add((v_h, v_d))
        #print(dic)
        dic = {}
        
    if dic_root != {}:                
        root_d = max(dic_root, key=dic_root.get)
        a.add((root_h, root_d))
        
    g_a = Graph(g.vertexs, a)
    
    if findonecycle(g_a.edges) == None:                
        print(g_a.edges)
        return g_a.edges
    else:
        c = set(findonecycle(g_a.edges))
        print('\ncycle:',c, '\n')
        g_c = contract(g, c, sigma, w, fm)
        y = chuliuedmonds(w, g_c, sigma, fm)
        
        # print(resolvecycle(y, c))
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


def dot(w, feature):
    f = np.zeros(w.shape)
    f[feature] += 1
    return np.dot(w, f)


def contract(g, c, sigma, w, fm):
    v_c = set()
    dic_c = {}
    root_has_child = 0  # in case ROOT already link to a v_d (dependent)

    represetation = sigma(g.vertexs, fm)  # a dict {(head, dependent):[1, 23,... 45],...}
    for i in c:
        v_c.add(i[0])
        v_c.add(i[1])
        dic_c[i[1]] = i[0] #key:dependent, value:head
     
    all_v_d = set(g.vertexs) - v_c
    g_c = set()

    # outside the cycle
    dic_root = {}
    dic_other = {}
    for i in g.edges:
        if i[0] in v_c or i[1] in v_c:
            pass
        else:
            if i[0].form == 'ROOT':
                root_h = i[0]
                root_has_child += 1
                r_score = dot(w, represetation[i])
                dic_root[i[1]] = r_score
            else:
                if i[1] not in dic_other:
                    dic_other[i[1]] = i[0]  # key: dependent, value: head
                else:
                    original = dot(w, represetation[(dic_other[i[1]], i[1])])
                    new = dot(w, represetation[(i[0], i[1])])
                    if new > original:
                        dic_other[i[1]] = i[0]
                        
    root_d = max(dic_root, key=dic_root.get)
    g_c.add((root_h, root_d))


    for e in dic_other:
        if e == root_d:
            pass
        else:
            g_c.add((dic_other[e], e))
            
    print(root_has_child)
    print('g_c_vd', g_c)

    
    # cycle -> v_d
    for d in all_v_d:
        dic_h = {}
        if d.form == 'ROOT':
            pass
        else:
            have_head = []
            for j in g_c:
                if d == j[1]:
                   have_head.append(j)
         
            for h in v_c:
                if root_has_child > 0:
                    if h.form == 'ROOT':
                        pass
                    else:
                        h_score = dot(w, represetation[(h, d)])
                        dic_h[h] = h_score
                else:
                    if h.form == 'ROOT':
                        root_has_child += 1
                    h_score = dot(w, represetation[(h, d)])
                    dic_h[h] = h_score
                    
            v_h = max(dic_h, key=dic_h.get)
            if have_head == None:
                g_c.add((v_h, d))
            else:
                original = dot(w, represetation[have_head[0]])
                new = dic_h[v_h]
                print(original, new)
                if new <= original:
                    pass
                else:
                    g_c.remove(have_head[0])
                    g_c.add((v_h, d))
                      

    print(root_has_child)
    print('g_c_out', g_c)                    
                    
    # cycle <- v_h
    for d in v_c:
        dic_h = {}
        for h in all_v_d:
            if root_has_child > 0:
                if h.form == 'ROOT':
                    pass
                else:
                    h_score = dot(w, represetation[(h, d)]) - dot(w, represetation[(dic_c[d], d)])
                    dic_h[h] = h_score
            else:
                if h.form == 'ROOT':
                    root_has_child += 1                    
                h_score = dot(w, represetation[(h, d)]) - dot(w, represetation[(dic_c[d], d)])
                dic_h[h] = h_score
        
        v_h = max(dic_h, key=dic_h.get)
        #print((v_h, v_d))
        g_c.add((v_h, d))
       
    print(root_has_child)
    print('g_c_in', g_c)
    
    return Graph(g.vertexs, g_c)

    
def resolvecycle(y, c):
    v_c = set()
    dic_c = {}
    del_edge = []
    for i in c:
        v_c.add(i[0])
        v_c.add(i[1])
        dic_c[i[1]] = i[0] #key:dependent, value:head

    result = y
    for e_y in y:
        v_d = e_y[1]
        if v_d in v_c:
            del_edge = [dic_c[v_d], v_d]
    for e_c in c:
        if e_c != set(del_edge):
            result.add(e_c)
        else:
            pass
    return result

    
    

#--------------test-----------------
if __name__ == '__main__':
    sents = reader("wsj_train.first-5k.conll06")
    for index_s in range(len(sents)):
        sent = sents[index_s].tokens
        vertexs = sent
        edges = get_edge(vertexs)
        #print()
        g = Graph(vertexs, edges)
        w = np.zeros(3000000)
        fm = FeatureMapping()
        chuliuedmonds(w, g, sigma,fm)

