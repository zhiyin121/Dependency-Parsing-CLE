import numpy as np

class chuliuedmonds(object):
    def __init__(self, score):
        self.score = score
        
    def cle(self):
        g_a = dict(enumerate(np.argmax(self.score, axis=0))) # {key: index_head; value: index_dependent}
        del g_a[0] # deleted ROOT -> ROOT
        c = self.findonecycle(g_a)
        if c == None:
            print(g_a)
            return g_a
        else:
            print('there is a confilct:', c)
        
    def findonecycle(self, g_a):
        circle = []
        def get_circle(dic, key): # Given the key(head), return a list of the cycle that it is in
            circle.append([key, dic[key]])
            key = dic[key]
            if key in dic and [key, dic[key]] not in circle:
                get_circle(dic, key)
            else:
                pass
            return circle # [[1(head), 2(dependent)],[2(head), 3(dependent)],...[5(head), 1(dependent)]]
            
        for i in range(1, len(g_a)): # Check the items of the dictionary one by one
            cycle = get_circle(g_a, i)
            if cycle != None:
                if cycle[0][0] == cycle[-1][-1]:
                    return cycle # until it find a cycle
                break
            else:
                continue
            circle = []
            
        

    def contract():
        pass
    
    def resolve():
        pass



score = np.array([[-10000, 9, 10, 9],
                  [-10000, -10000, 20, 3],
                  [-10000, 30, -10000, 30],
                  [-10000, 11, 0, -10000]])
print(score)
#column = np.arange(start=0, stop=score.shape[1], step=1)
#row = np.argmax(score, axis=0)
#g_a = np.array([column, row]).T


aaa = chuliuedmonds(score)
aaa.cle()

'''
g_a = dict(enumerate(np.argmax(score, axis=0)))
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




