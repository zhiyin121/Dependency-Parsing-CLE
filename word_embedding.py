from tools import reader
import numpy as np


def get_edge(vertexs):  
    edges = set()
    for i in vertexs:
        for j in vertexs:
            if i != j and j.form != 'ROOT': ##remember to set the same rule in the model as well
                edges.add((i, j))
    return edges


def word_embedding(filename):
    sents = reader(filename)
    vocab_dic = {}
    pos_dic = {}
    vocab_count = {}
    pos_count = {}
    for index_s in range(len(sents)):
        tokens = sents[index_s].tokens
        for index_t in range(len(tokens)):
            dependent = tokens[index_t]
            # build vocabulary
            if vocab_dic == {}:
                vocab_dic[str(dependent)] = 0
            else:
                if str(dependent) in vocab_dic:  #!!!did not disdinguish
                    pass
                else:
                    vocab_dic[str(dependent)] = max(vocab_dic.values()) + 1
                    
            # build pos dictionary
            if pos_dic == {}:
                pos_dic[str(dependent.pos)] = 0
            else:
                if str(dependent.pos) in pos_dic:
                    pass
                else:
                    pos_dic[str(dependent.pos)] = max(pos_dic.values()) + 1

            # count how many times a specific word appears
            if str(dependent) in vocab_count:
                vocab_count[str(dependent)] += 1
            else:
                vocab_count[str(dependent)] = 1
                
            # count how many times a specific pos appears
            if str(dependent.pos) != 'ROOT':
                if str(dependent.pos) in pos_count:
                    pos_count[str(dependent.pos)] += 1
                else:
                    pos_count[str(dependent.pos)] = 1                

                
    #print(len(vocab_dic))
    #print(len(pos_dic))
    
    word_embed = np.zeros((len(vocab_dic), len(vocab_dic)))
    #print(word_embed.shape)
    '''
                head1   head2   head3 ...
    dependent1 
    dependent2
    dependent3
    ...
    '''
    '''
    word_embed[1][1] += 2
    print(word_embed[1][1])
    '''
    pos_embed = np.zeros((len(pos_dic), len(pos_dic)))
    #print(pos_embed.shape)

    
    for index_s in range(len(sents)):
        tokens = sents[index_s].tokens
        edges = get_edge(tokens)
        for edge in edges:
            dependent = edge[1]
            head_p = edge[0]
            head_g = tokens[int(dependent.head)]
            if str(dependent) != 'ROOT':
                if head_p == head_g:
                    word_embed[vocab_dic[str(dependent)]][vocab_dic[str(head_p)]] += 2
                    pos_embed[pos_dic[str(dependent.pos)]][pos_dic[str(head_p.pos)]] += 1
                else:
                    word_embed[vocab_dic[str(dependent)]][vocab_dic[str(head_p)]] += 0.2
                    pos_embed[pos_dic[str(dependent.pos)]][pos_dic[str(head_p.pos)]] += 0.1

        '''
        for index_t in range(len(tokens)):
            dependent = tokens[index_t]
            if str(dependent) != 'ROOT':
                head = tokens[int(dependent.head)]
                word_embed[vocab_dic[str(dependent)]][vocab_dic[str(head)]] += 1
        '''
    for word in vocab_dic:
        word_embed[vocab_dic[word]] = word_embed[vocab_dic[word]]/vocab_count[word]
    for pos in pos_dic:
        if pos != 'ROOT':
            pos_embed[pos_dic[pos]] = pos_embed[pos_dic[pos]]/pos_count[pos]
    
    return vocab_dic, word_embed, pos_dic, pos_embed


#-----------------test---------------
if __name__ == '__main__':
    vocab_dic, word_embed, pos_dic, pos_embed = word_embedding("tiger-2.2.dev.conll06.gold")
    print('finised!')

    result = ''
    for i in range(len(pos_dic)):
        print(list(pos_embed[i]))

    np.savetxt("word_embedding_vocab.txt",word_embed)
    np.savetxt("word_embedding_pos.txt",pos_embed)
    
              
    result = ''
    for i in range(len(pos_dic)):
        print(list(pos_embed[i]))
        '''
        result += str(list(pos_embed[i])) + '\n'
        with open("word_embedding_pos.txt", 'w', encoding='utf-8') as w:
            w.write(result)
            '''
    
  
