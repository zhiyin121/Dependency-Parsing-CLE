class Token(object):
    def __init__(self, token):
        self.form = token[1]
        self.pos = token[3]
        self.head = token[6]
        self.label = token[7]
        self.line = token

    def __repr__(self):
        return self.form

 
class Sentence(object):
    def __init__(self, list_of_token):
        self.tokens = list_of_token

    #def __repr__(self):
    #    return self.tokens


def reader(filename):
    file = open(filename, 'r', encoding='utf-8')
    token_list = list(file)
    line_count = len(token_list)
    
    sents = []; sent = [Token([0, 'ROOT', '_', 'ROOT', '_', '_', None, '_', '_', '_'])]    
    for i in range(line_count):
        token = token_list[i].strip('\n').split('\t')
        
        if token[0] != '':
            sent.append(Token(token))
        else:
            #print(sent)
            sents.append(Sentence(sent))
            sent = [Token([0, 'ROOT', '_', 'ROOT', '_', '_', None, '_', '_', '_'])]
            
    return sents  #[<['ROOT','yeah', '!']>, <[<''>,...]>]


def writer(filename, sentences):
    content = ''
    w = open(filename, 'w', encoding='utf-8')
    for sentence in sentences: 
        for token in sentence.tokens[1:]:
            line = token.line
            content += '\t'.join(line) + '\n'
        content += '\n'
    w.write(content)
            
          
'''
#-------testing------

sents = reader("devconll06pred.txt")
#print(sents[0].tokens[1].form)

w = writer('test_write.txt', sents)
'''

class Evaluate(object):
    def __init__(self, pred_filename, gold_filename):
        self.pred_filename = pred_filename
        self.gold_filename = gold_filename
        self.head = 'head'
        
    def get_sentences(self):
        pred = reader(self.pred_filename)
        gold = reader(self.gold_filename)
        return pred, gold

    def get_token_elements(self):
        pred, gold = self.get_sentences()

        head_list = []
        label_list = []
        for i in range(len(pred)):
            for j in range(1, len(pred[i].tokens)):
                head_list.append((pred[i].tokens[j].head, gold[i].tokens[j].head))
                label_list.append((pred[i].tokens[j].label, gold[i].tokens[j].label))
        return  head_list, label_list #a list of tuples (pred_head, gold_head)
                
    def uas(self):
        head_list = self.get_token_elements()[0]
        correct_head = 0
        number_token = len(head_list)
        for i in head_list:
            if i[0] == i[1]:
                correct_head += 1
        ua_score = round(correct_head/number_token, 4)
        print('correct_head:', correct_head)
        print('number_token:', number_token)
        print('uas:', ua_score)
        return ua_score

    def las(self):
        head_list, label_list = self.get_token_elements()
        correct_head_label = 0
        number_token = len(head_list)
        for i in range(len(head_list)):
            if head_list[i][0] == head_list[i][1] and label_list[i][0] == label_list[i][1]:
                correct_head_label += 1
        la_score = round(correct_head_label/number_token, 4)
        print('correct_head&label:', correct_head_label)
        print('number_token:', number_token)
        print('las:', la_score)
        return la_score


'''
#-------testing------            
a= Evaluate('wsj_dev.conll06.pred', 'wsj_dev.conll06.gold')
a.uas()
a.las()
'''































