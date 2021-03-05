from tools import reader
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
        """
        if hform != 'ROOT':
            grand = sentence[int(head.head)]
            gform = grand.form
            gpos = grand.pos
        else:
            gform = '_NULL_'
            gpos = '_NULL_'
"""
        if head == sentence[int(dependent.head)]:
            gold_feature[pair] = {'dform':dform, 'dpos':dpos, 'hform':hform, 'hpos':hpos}#, 'gform':gform, 'gpos':gpos}     
        else:
            nega_feature[pair] = {'dform':dform, 'dpos':dpos, 'hform':hform, 'hpos':hpos}#, 'gform':gform, 'gpos':gpos}
        
    return gold_feature, nega_feature   #format: {(head_1, dependent_1):{{'dform':'dependent', 'dpos':'XX'},{},{}},
                                        #         (head_2, dependent_2):{{'dform':'dependent', 'dpos':'XX'},{},{}}


def get_feature_test(sentence):
    feature = {}
    all_edges = get_edge(sentence)
    
    for pair in all_edges:
        dependent = pair[1]
        dform = dependent.form
        dpos = dependent.pos        
        
        head = pair[0]
        hform = head.form
        hpos = head.pos
        feature[pair] = {'dform':dform, 'dpos':dpos, 'hform':hform, 'hpos':hpos}
        
    return feature


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

'''        
start = time.time()
sents = reader("wsj_train.first-5k.conll06")
print(len(sents))

dic_f = {} #{feature string: index, ...}
dic_e = {} #{edge tuple: vector representation, ...}

for index_s in range(len(sents)):
    sent = sents[index_s].tokens
    dic_features, dic_edges = get_dict(dic_f, sent)
    print(dic_edges)
'''

class FeatureMapping(object):
    def __init__(self):
        self.mapping = {}
        self.frozen = False

    def get_dict(self, sent): #dic_f = {} -> {feature string: index, ...}
                              #dic_e = {} -> {edge tuple: vector representation, ...}
        dic_e = {}
        if not self.frozen:
            gold_feature, nega_feature = get_feature(sent)
                    
            for edge in gold_feature:  #Gold
                templates = get_template(gold_feature[edge])
                indexs = []
                    
                for tem in templates:
                    tem = ','.join(tem)

                    if self.mapping == {}:
                        self.mapping[str(tem)] = 0
                        index = 0
                    else:
                        if str(tem) in self.mapping:
                            index = self.mapping[str(tem)]
                        else:
                            index = max(self.mapping.values()) + 1
                        self.mapping[str(tem)] = index
                    indexs.append(index)

                if edge in dic_e:
                    pass  ##could be add index of the new features, instead of ignore it
                else:
                    dic_e[edge] = indexs

            for edge in nega_feature:  #Negative
                templates = get_template(nega_feature[edge])

                for tem in templates:
                    tem = ','.join(tem)

                    if self.mapping == {}:
                        self.mapping[str(tem)] = 0
                    else:
                        if str(tem) in self.mapping:
                            pass
                        else:
                            self.mapping[str(tem)] = max(self.mapping.values()) + 1
                        
                dic_e[edge] = []
        else:
            feature = get_feature_test(sent)
            for edge in feature:
                templates = get_template(feature[edge])
                indexs = []
                    
                for tem in templates:
                    tem = ','.join(tem)
                    if str(tem) in self.mapping:
                        index = self.mapping[str(tem)]
                        indexs.append(index)
                dic_e[edge] = indexs
                
        return dic_e

