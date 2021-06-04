import torch
from torch._C import device
import dataset, model, tools
import os, pickle
from torch.autograd import Variable
from dependency_decoding import chu_liu_edmonds # requires https://github.com/andersjo/dependency_decoding


# load data
if not os.path.exists('dictionary.de.pkl'):
    raise NotImplementedError
else:
    dics = pickle.load(open('dictionary.de.pkl', 'rb'))
    dic_form = dics['dic_form']
    dic_pos = dics['dic_pos']
    dic_label = dics['dic_label']

testset = dataset.Dataset(dic_form, dic_pos, dic_label)
testset.load('german/dev/tiger-2.2.dev.conll06.blind', training=False)

batch_size = 1
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# load embedding
if not os.path.exists('embedding.pkl'):
    raise NotImplementedError
else:
    embedding = pickle.load(open('embedding.de.pkl', 'rb'))
    pretrained_word_embeddings, pretrained_pos_embeddings = embedding

# init model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load(open('model.de.pt', 'rb'), map_location=device)

pred = []
for batch_ndx, sample in enumerate(test_loader):
    seq_len = len(sample['form'])
    if seq_len <= 1: 
        continue

    gold_tree = torch.LongTensor(sample['tree'])
    arc_target = Variable(gold_tree, requires_grad=False)
    labels_target = torch.LongTensor(sample['label'])
    
    sequence = torch.LongTensor(seq_len, 2)
    # sequence[:,2] = gold_tree
    sequence[:,1] = torch.LongTensor(sample['pos'])
    sequence[:,0] = torch.LongTensor(sample['form'])
    sequence_var = Variable(sequence)

    # prepare GPU
    if torch.cuda.is_available():
        arc_target = arc_target.cuda()
        sequence_var = sequence_var.cuda()
        
    adj_mat, _ = model(sequence_var)
    # adj_mat = F.softmax(torch.t(adj_mat))
    arc_pred = torch.t(adj_mat)

    _, pred_single = torch.max(arc_pred, 1)
    # print(arc_acc)

    if torch.cuda.is_available():
        adj_mat = adj_mat.cpu()
        pred_single = pred_single.cpu()

    adj_mat = adj_mat.data.double().numpy()
    # print(adj_mat)
    pred_single2, _ = chu_liu_edmonds(adj_mat.T)
    pred_single2 = [0] + pred_single2[1:]
    pred_single2 = torch.LongTensor(pred_single2)

    # print(pred_single, pred_single2)
    # print(pred_single, gold_tree)
    # For labels
    if torch.cuda.is_available():
        sequence_with_label = torch.cuda.LongTensor(seq_len, 3)
    else:
        sequence_with_label = torch.LongTensor(seq_len, 3)

    sequence_with_label[:,2] = torch.LongTensor(pred_single2)
    sequence_with_label[:,1] = torch.LongTensor(sample['pos'])
    sequence_with_label[:,0] = torch.LongTensor(sample['form'])
    sequence_var = Variable(sequence_with_label)

    adj_mat, predicted_labels = model(sequence_var)

    pred_labels = [0]*len(predicted_labels)
    for i in range(len(predicted_labels)):
        _, pred_labels[i] = predicted_labels[i].max(0)
    pred_labels = torch.LongTensor(pred_labels)
    pred.append((pred_single2.numpy(), pred_labels.numpy()))


sentences = tools.reader('german/dev/tiger-2.2.dev.conll06.blind')

for s, p in zip(sentences, pred):
    for i, token in enumerate(s.tokens):
        token.line[6] = str(p[0][i])
        token.line[7] = dic_label.i2w[p[1][i]]
        pass

tools.writer('german/dev/tiger-2.2.dev.conll06.pred2', sentences)