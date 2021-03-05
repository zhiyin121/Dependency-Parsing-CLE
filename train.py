from tools import reader, writer
from algorithm_revised import Graph, sigma, chuliuedmonds
from feature_revised import FeatureMapping
import numpy as np

def get_edge(vertexs):  
    edges = set()
    for i in vertexs:
        for j in vertexs:
            if i != j and j.form != 'ROOT': ##remember to set the same rule in the model as well
                edges.add((i, j))
    return edges


w = np.zeros(3000000)
print(type(w))
sents = reader("wsj_train.first-5k.conll06")
for index_s in range(len(sents)):
    sent = sents[index_s].tokens
    vertexs = sent
    edges = get_edge(vertexs)
    #print()
    g = Graph(vertexs, edges)
    
    fm = FeatureMapping()

    pred_y = chuliuedmonds(w, g, sigma, fm)
    print(pred_y, '\n')

