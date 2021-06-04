import tools
import torch
from gensim.models import Word2Vec


class Dictionary():
    def __init__(self):
        self.w2i = {'<unk>':0}
        self.i2w = ['<unk>']

    def add_word(self, word):
        self.w2i[word] = len(self.i2w)
        self.i2w.append(word)

    def get_index(self, word, training=True):
        if word in self.w2i:
            return self.w2i[word]
        elif training:
            self.add_word(word)
            return self.w2i[word]
        else:
            return 0


class Dataset():
    def __init__(self, dic_form, dic_pos, dic_label):
        self.dic_form = dic_form
        self.dic_pos = dic_pos
        self.dic_label = dic_label
        self.forms = []
        self.poss = []
        self.labels = []
        self.tree = []

    def load(self, filename, training=True):
        sentences = tools.reader(filename)
        for s in sentences:
            form = []
            pos = []
            label = []
            tree = [0]
            sentence_len = len(s.tokens)
            for token in s.tokens:
                word_id = int(token.id)
                if word_id != 0:
                    if token.head == '_':
                        head = 0
                    else:
                        head = int(token.head)
                    tree.append(head)
                
                label.append(self.dic_label.get_index(token.label, training))
                form.append(self.dic_form.get_index(token.form, training))
                pos.append(self.dic_pos.get_index(token.pos, training))
                
            self.forms.append(form)
            self.poss.append(pos)
            self.labels.append(label)
            self.tree.append(tree)
      
    def __len__(self):
        return len(self.forms)

    def __getitem__(self, index):
        return {
            'form': self.forms[index],
            'pos': self.poss[index],
            'label': self.labels[index],
            'tree': self.tree[index]
        }


def pretrain_word_embeddings(data, len_word_embed, len_pos_embed, dic_form, dic_pos):
    '''
    corpus_words: [['I', 'like', 'custard'],...]
    corpus_pos: [['NN', 'VB', 'PRN'],...]
    '''

    corpus_words = []
    corpus_pos = []

    for s in range(len(data)):
        # print(data[s]['pos'])
        words = [dic_form.i2w[i] for i in data[s]['form']]
        pos_s = [dic_pos.i2w[i] for i in data[s]['pos']]

        corpus_words.append(words)
        corpus_pos.append(pos_s)

    # pre-train word and pos embeddings. These will be starting points for our learnable embeddings
    word_embeddings_gensim = Word2Vec(corpus_words, size=len_word_embed, window=5, min_count=1, workers=8)
    pos_embeddings_gensim = Word2Vec(corpus_pos, size=len_pos_embed, window=5, min_count=1, workers=8)

    # initialise the embeddings. The tensors are still empty
    pretrained_word_embeddings = torch.FloatTensor(len(dic_form.i2w)+1, len_word_embed)
    pretrained_pos_embeddings = torch.FloatTensor(len(dic_pos.i2w)+1, len_pos_embed)

    # fill the tensors with the pre-trained embeddings
    for word in dic_form.w2i.keys():
        if word != '<unk>':
            idx = dic_form.w2i[word]
            pretrained_word_embeddings[idx, :] = torch.from_numpy(word_embeddings_gensim[word])

    for pos in dic_pos.w2i.keys():
        if pos != '<unk>':
            idx = dic_pos.w2i[pos]
            pretrained_pos_embeddings[idx, :] = torch.from_numpy(pos_embeddings_gensim[pos])

    return pretrained_word_embeddings, pretrained_pos_embeddings

