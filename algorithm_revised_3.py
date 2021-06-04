# re-write contract

import numpy as np
from tools import reader
from feature_revised import FeatureMapping
from word_embedding import word_embedding
import time
import json


def get_edge(vertexs):
    edges = set()
    for i in vertexs:
        for j in vertexs:
            if i != j:
                if type(i) is not tuple and type(j) is not tuple:
                    if j.form != 'ROOT':
                        edges.add((i, j))
                else:
                    if type(i) is tuple:
                        for word in i:
                            edges.add((word, j))
                    if type(j) is tuple:
                        for word in j:
                            edges.add((i, word))
    return edges


class Graph(object):
    def __init__(self, vertexs, edges):  # !!!V:<v_h, v_d>, E: representation of edges
        self.vertexs = vertexs
        self.edges =  edges


def sigma(edges, filename):
    with open(filename,'r') as load_f:
        load_dict = json.load(load_f)

    dic_e = {}
    for edge in edges:
        if str(edge) in load_dict:
            dic_e[edge] = load_dict[str(edge)]
        else:
            dic_e[edge] = []
    
    return dic_e


def sigma_2(edges):
    
    represetation = {}
    for edge in edges:
        if edge[0].form in vocab_dic and edge[1].form in vocab_dic:
            represetation[edge] = word_embed[vocab_dic[str(edge[1])]][vocab_dic[str(edge[0])]] + \
                                  pos_embed[pos_dic[str(edge[1].pos)]][pos_dic[str(edge[0].pos)]]
        else:
            represetation[edge] = 0.5
    return represetation


def chuliuedmonds(w, g, sigma, fm, counts):
    
    #represetation = sigma(g.edges, "test_save_representation_1.json")  # a dict {(head, dependent):[1, 23,... 45],...}
    represetation = sigma_2(g.edges)
    #print(represetation)

    #print('g.edges:\n', g.edges)

    # get all vertexs appear in g.edges
    all_vertexs = set()
    for edge in g.edges:
        all_vertexs.add(edge[0])
        all_vertexs.add(edge[1])
    #print('all_vertexs:\n', all_vertexs)

    # find all best-scoring heads
    a = set()
    for vertex in all_vertexs:
        if vertex.form != 'ROOT':
            head_of_vertex = {}
            for edge in g.edges:
                if edge[1] == vertex:
                    head_of_vertex[edge[0]] = represetation[edge]                    
            highest_head = max(head_of_vertex, key=head_of_vertex.get)
            a.add((highest_head, vertex))
        else: # (ROOT, )
            child_of_vertex = {}
            for edge in g.edges:
                if edge[0] == vertex:
                    child_of_vertex[edge[1]] = represetation[edge]
            highest_child = max(child_of_vertex, key=child_of_vertex.get)
            a.add((vertex, highest_child))
            
    # check if ther is a cycle    
    g_a = Graph(g.vertexs, a)
    #print('g_a.edges:', g_a.edges)
    c = findonecycle(g_a.edges)  # a set of vertexs that in the cycle
    
    if c == None:                
        #print(g_a.edges)
        return g_a.edges
    else:
        #print('\ncycle:', c, '\n')
        g_c = contract(g, c, sigma, w, fm)
        
        counts += 1
        #print(counts)
        if counts > 10:
            return resolvecycle(g_a.edges, c)
        else:
            y = chuliuedmonds(w, g_c, sigma, fm, counts)
                
            # print(resolvecycle(y, c))
            return resolvecycle(y, c)
        

def findonecycle(g_a):
    
    def get_next(g_a, dependent):
        next_pair = []
        for i in g_a:
            if dependent == i[0]:
                next_pair = i
        if next_pair in lst:
            pass
        else:
            if next_pair != []:
                if next_pair[1] != lst[0][0]:
                    lst.append(next_pair)
                    get_next(g_a, next_pair[1])
                else:
                    lst.append(next_pair)
        return lst
    # print(g_a)
    for i in g_a:
        lst = [i]
        get_next(g_a, i[1])
        if lst[0][0] == lst[-1][1]:
            v_c = set()
            for edge in lst:
                v_c.add(edge[0])
                v_c.add(edge[1])
            return v_c  # a set of vertexs that in the cycle
    

def dot(w, feature):
    # print(feature)
    f = np.zeros(w.shape)
    f[feature] += 1
    return np.dot(w, f)


def contract(g, c, sigma, w, fm):
    # get vertexs out of cycle
    vertexs = set()
    v_d = set()
    for vertex in g.vertexs:
        if vertex in c:
            pass
        else:
            vertexs.add(vertex)
            v_d.add(vertex)

    # combine the notes of cycle into a new note
    new_node = tuple(c)
    
    # add the new note, get edges except those in cycle
    v_d.add(new_node)
    edges = get_edge(v_d)
    #print(edges)

    # store score for each edges(all)
    all_edges = get_edge(g.vertexs)
    represetation = sigma_2(all_edges) # a dict {(head, dependent):0.22, .....}
    
    g_c = set()
    
    for edge in edges:
        if type(edge[0]) is tuple:
            c_edges_d = {}
            for head in edge[0]:
                c_edges_d[(head, edge[1])] = represetation[(head, edge[1])]
            highest_head = max(c_edges_d, key=c_edges_d.get)
            g_c.add(highest_head)
            
        elif type(edge[1]) is tuple:
            d_edges_c = {}
            for child in edge[1]:
                d_edges_c[(edge[0], child)] = represetation[(edge[0], child)]
            highest_child = max(d_edges_c, key=d_edges_c.get)
            g_c.add(highest_child)
        else:
            g_c.add(edge)
            
    #print('after contract', g_c)
    return Graph(list(vertexs), g_c)

    
def resolvecycle(y, c):
    #print('y:\n', y)
    #print('c:\n', c)
    
    edge_c = set()
    for i in c:
        for j in c:
            if i != j:
                edge_c.add((i, j)) # key:dependent, value:head

    #print('edge_c_before:', edge_c)
    
    common_child = set()
    for edge in y:
        if edge[1] in c:
            common_child.add(edge[1])
    #print('common_child', common_child)

    vc_set = set()
    for vc in common_child:
        for edge in edge_c:
            if edge[0] == vc and edge[1] not in common_child:
                vc_set.add(edge)
           
    result = y.union(vc_set)
    #print('result:\n', result)

    return result

    
    

#--------------test-----------------
if __name__ == '__main__':
    sents = reader("tiger-2.2.dev.conll06.blind")
    fm = FeatureMapping()
    w = np.random.random(3000000)
    w = open('test_result.txt', 'a+', encoding='utf-8')
    vocab_dic, word_embed, pos_dic, pos_embed = word_embedding("tiger-2.2.train.conll06")
    
    for index_s in range(len(sents)):
        sent = sents[index_s].tokens
        
        vertexs = sent
        edges = get_edge(vertexs)
        #print()
        g = Graph(vertexs, edges)
        print('Start -', index_s)
        start = time.time()
        
        counts = 1
        result = chuliuedmonds(w, g, sigma, fm, counts)
        
        #print('result:\n', result, '\n')

        voc_dic = dict(enumerate(sent))
        voc = {v : k for k, v in voc_dic.items()}
        #print('voc', voc)
        
        word_lst = [0]*(max(voc.values())+1)
        #print(len(word_lst))
        for word in voc:
            word_lst[voc[word]] = word
        #print('word_list:', word_lst, '\n')

        
        head_lst = [0]*(max(voc.values())+1)
        head_lst[0] = '__NULL__'
        
        for pair in result:
            head = pair[0]
            child = pair[1]

            head_i = word_lst.index(child)
            if head.form == 'ROOT':
                head_lst[head_i] = 0
            else:
                head_lst[head_i] = word_lst.index(head)
        del head_lst[0]
        print('head_lst:', head_lst)

        del word_lst[0]
        pred = ''
        for word_index in range(len(word_lst)):
            word = word_lst[word_index]
            #print(word.form, word.id)
            pred += word.id + '\t' + word.form + '\t' + \
                    word.lemma + '\t' + word.pos + '\t' + \
                    '_' + '\t' + '_' + '\t' + \
                    str(head_lst[word_index]) + '\t' + '_' + \
                    '\t' + '_' + '\t' + '_' + '\n'
        #print(pred)
        w.write(pred+'\n')
                
        end = time.time()
        print('TIME:', end-start)
    w.close()
