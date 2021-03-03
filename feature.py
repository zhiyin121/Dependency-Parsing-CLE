from tools import reader, writer
from itertools import combinations
import time

def get_edge(vertexs):  
    edges = set()
    for i in vertexs:
        for j in vertexs:
            if i != j and j.form != 'ROOT': ##remember to set the same rule in the model as well
                edges.add((i, j))
    return edges

    
def get_feature(sentence):
    nega_feature = {}
    gold_feature = {}
    
    all_edges = get_edge(sentence)
    
    for pair in all_edges:
        dependent = pair[1]
        dform = dependent.form
        dpos = dependent.pos        
        
        head = pair[0]
        hform = head.form
        hpos = head.pos
        
        if hform != 'ROOT':
            grand = sentence[int(head.head)]
            gform = grand.form
            gpos = grand.pos
        else:
            gform = '_NULL_'
            gpos = '_NULL_'

        if head == sentence[int(dependent.head)]:
            gold_feature[pair] = {'dform':dform, 'dpos':dpos, 'hform':hform, 'hpos':hpos, 'gform':gform, 'gpos':gpos}     
        else:
            nega_feature[pair] = {'dform':dform, 'dpos':dpos, 'hform':hform, 'hpos':hpos, 'gform':gform, 'gpos':gpos}
        
    return gold_feature, nega_feature   #format: {(head_1, dependent_1):{{'dform':'dependent', 'dpos':'XX'},{},{}},
                                        #         (head_2, dependent_2):{{'dform':'dependent', 'dpos':'XX'},{},{}}
        

def get_template(feature):  #format: ({'dform':dform, 'dpos':dpos}, {'hform':hform, 'hpos':hpos}, {'gform':gform, 'gpos':gpos})
    uni_g = []
    for uni in feature:
        f = str(uni) + ':' + str(feature[uni])
        uni_g.append(f)
        
    mul_g = []
    n = 2  ##get n_gram feature up to n #hyperparameter#
    for i in range(1, n+1):  
        mul_g.extend(combinations(uni_g, i))
    
    #print(mul_g)
    return mul_g
        

def get_dict(dic_f, dic_e, sents):    #dic_f = {} -> {feature string: index, ...}
                                     #dic_e = {} -> {edge tuple: vector representation, ...}
    for index_s in range(len(sents)):
        if index_s%10 == 0:
            print(index_s)
            end = time.time()
            print(end-start)
        sentence = sents[index_s].tokens
        gold_feature, nega_feature = get_feature(sentence)
            
        for edge in gold_feature:  #Gold
            templates = get_template(gold_feature[edge])
            indexs = []
            
            for tem in templates:
                tem = ','.join(tem)

                if dic_f == {}:
                    dic_f[str(tem)] = 0
                    index = 0
                else:
                    if str(tem) in dic_f:
                        index = dic_f[str(tem)]
                    else:
                        index = max(dic_f.values()) + 1
                    dic_f[str(tem)] = index
                indexs.append(index)

            if edge in dic_e:
                pass  ##could be add index of the new features, instead of ignore it
            else:
                dic_e[edge] = indexs


        for edge in nega_feature:  #Negative
            templates = get_template(nega_feature[edge])

            for tem in templates:
                tem = ','.join(tem)

                if dic_f == {}:
                    dic_f[str(tem)] = 0
                else:
                    if str(tem) in dic_f:
                        pass
                    else:
                        dic_f[str(tem)] = max(dic_f.values()) + 1
                    
            dic_e[edge] = []
            
            #print(dic_e)    
    return dic_f, dic_e
        
start = time.time()
sents = reader("wsj_train.first-5k.conll06")
print(len(sents))
dic_f = {} #{feature string: index, ...}
dic_e = {} #{edge tuple: vector representation, ...}
dic_features, dic_edges = get_dict(dic_f, dic_e, sents)
print(len(dic_features), len(dic_edges))

class FeatureMapping(object):
    def __init__(self, feature):
        self.feature = feature
        
        
    def map(self):
        pass
            


    def frozen():
        pass
