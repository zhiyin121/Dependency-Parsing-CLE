from tools import reader, writer

sents = reader("devconll06pred.txt")
print(sents[0].tokens[1].form, sents[0].tokens[1].pos, sents[0].tokens[int(sents[0].tokens[1].head)])

class FeatureMapping(object):
    def __init__(self, string, integer):
        self.dic = dic
        
        
    def map(self):
        pass
            


    def frozen():
        pass
