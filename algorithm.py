import numpy as np
import copy

class chuliuedmonds(object):
    def __init__(self, graph, sigma):
        self.g = graph
        self.sigma = sigma

        
    def cle(self, graph, sigma):
        print(graph)
        g_a = []
        g_a_old_1 = dict(enumerate(np.argmax(graph, axis=0))) # [index_head, index_dependent]
        g_a_old_2 = dict(enumerate(np.argmax(graph, axis=1)))
        del g_a_old_1[0] # deleted ROOT -> ROOT
        for key in g_a_old_1:
            if  key != 0:
                g_a.append([g_a_old_1[key], key])
        del g_a_old_2[0] # deleted ROOT -> ROOT
        for key in g_a_old_2:
            if [key, g_a_old_2[key]] not in g_a and g_a_old_2[key] != 0:
                g_a.append([key, g_a_old_2[key]])
        
            
        print(g_a)
        
        c = self.findonecycle(g_a)
        print(c)
        
        if c == None:
            
            return g_a
        else:
            print('there is a confilct:', c)
            g_c= self.contract(graph, c, sigma)
            y = self.cle(g_c, sigma)
            return self.resolvecycle(y, c)
            
        
    def findonecycle(self, g_a):
        
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
                    return lst.append(next_pair)
          
            #print(lst)
            return lst

        for i in g_a:
            lst = [i]
            get_next(g_a, i[1])
            if lst[0][0] == lst[-1][1]:
                #print(lst)
                return lst


    def contract(self, graph, c, sigma):
        g_c = copy.deepcopy(graph)
        v_c = set()  # all nodes in cycle
        score_c = 0
        for i in c:
            #g_c[i[0],i[1]] = -10000
            v_c.add(i[0])
            score_c += graph[i[0],i[1]]

        for node in range(graph.shape[0]): 
            if node not in v_c:
                for head in v_c:
                    pass # cannot store <head, node> when head has child in v_c
                dic = {}
                for dependent in v_c:
                    for i in c:
                        if i[1] == dependent:
                            h_d = i[0]
                        else:
                            pass
                    new_score = graph[node, dependent] + score_c - graph[h_d, dependent]
                dic[dependent] = new_score
                chosed_dependent = max(dic, key=dic.get)
                g_c[node, chosed_dependent] = dic[chosed_dependent]
                for i in c:
                    if i[1] == chosed_dependent:
                        h_chosed_dependent = i[0]
                g_c[h_chosed_dependent, chosed_dependent] /= 1.5
        return g_c

    
    def resolvecycle(self, y, c):
        print('answer', y)
        print('cycle', c)
        for pair in c:
            if pair in y:
                pass
            else:
                y.append(pair)
        print('result', y)       
        return y


#g = ['ROOT', 'John', 'saw', 'Mary']
g = np.array([[0, 1, 0], [1, 2, 3], [2, 4, 2], [0, 2, 1]])

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def sigma(head, depentdent):
    result = round(cosine_similarity(head, depentdent), 2)*100
    return result

def get_graph(g):
    shape = [len(g), len(g)]
    graph = np.zeros(shape)
    for h in range(len(g)):
        for d in range(len(g)):
            if d == 0 or h == d:
                graph[h, d] = -10000
            else:
                graph[h, d] = sigma(g[h], g[d])
    return graph

#graph = np.array([[-10000, 9, 10, 9],
#                  [-10000, -10000, 20, 3],
#                  [-10000, 30, -10000, 30],
#                  [-10000, 11, 0, -10000]])
#print(graph)
#column = np.arange(start=0, stop=graph.shape[1], step=1)
#row = np.argmax(graph, axis=0)
#g_a = np.array([column, row]).T

graph = get_graph(g)
aaa = chuliuedmonds(graph, sigma)
aaa.cle(graph, sigma)


'''
g_a = dict(enumerate(np.argmax(graph, axis=0)))
print(g_a)

test = {0: 0, 1: 2, 2: 3, 3: 2, 4: 5, 5: 4, 6:2}
circle = []
def get_value(dic, key):
    circle.append([key, dic[key]])
    key = dic[key]
    if key in dic and [key, dic[key]] not in circle:
        get_value(dic, key)
    else:
        print(circle)
        return circle
        
for i in range(len(test)):
    get_value(test, i)
    circle = []
'''




