import torch
from dependency_decoding import chu_liu_edmonds # requires https://github.com/andersjo/dependency_decoding
import dataset


CUDA = True

def UAS_score(pred_tree, gold_tree):
    length = len(pred_tree)
    count = (pred_tree.cpu() == gold_tree.cpu()).sum().item()
    return count, length


def LAS_score(pred_tree, gold_tree, pred_labels, gold_labels):
    length = len(pred_tree)
    count = 0
    for i in range(length):
        if pred_tree[i] == gold_tree[i]:
            if pred_labels[i].cpu().item() == gold_labels[i]:
                count += 1
    return count, length


def test(model, sample):
    seq_len = len(sample['form'])
    if seq_len <= 1: 
        return

    gold_tree = torch.LongTensor(sample['tree'])
    labels_target = torch.LongTensor(sample['label'])
    
    sequence = torch.LongTensor(seq_len, 2)
    # sequence[:,2] = gold_tree
    sequence[:,1] = torch.LongTensor(sample['pos'])
    sequence[:,0] = torch.LongTensor(sample['form'])

    # Move to GPU
    if torch.cuda.is_available() and CUDA:
        gold_tree = gold_tree.cuda()
        sequence = sequence.cuda()
        
    adj_mat, _ = model(sequence)
    # adj_mat = F.softmax(torch.t(adj_mat))
    arc_pred = torch.t(adj_mat)

    _, pred_single = torch.max(arc_pred, 1)

    if torch.cuda.is_available() and CUDA:
        adj_mat = adj_mat.cpu()
        pred_single = pred_single.cpu()

    adj_mat = adj_mat.data.double().numpy()

    pred_single2, _ = chu_liu_edmonds(adj_mat.T)
    pred_single2 = [0] + pred_single2[1:]
    pred_single2 = torch.LongTensor(pred_single2)

    # Labels of arcs
    if torch.cuda.is_available() and CUDA:
        sequence_with_label = torch.cuda.LongTensor(seq_len, 3)
    else:
        sequence_with_label = torch.LongTensor(seq_len, 3)

    sequence_with_label[:,2] = torch.LongTensor(pred_single)
    sequence_with_label[:,1] = torch.LongTensor(sample['pos'])
    sequence_with_label[:,0] = torch.LongTensor(sample['form'])

    adj_mat, predicted_labels = model(sequence_with_label)

    pred_labels = [0]*len(predicted_labels)
    for i in range(len(predicted_labels)):
        _, pred_labels[i] = predicted_labels[i].max(0)

    pred_tree = torch.LongTensor(pred_single).cpu()
    uas_chuliu = UAS_score(pred_single2, gold_tree)
    uas_score = UAS_score(pred_tree, gold_tree)
    las_score = LAS_score(pred_tree.cpu(), gold_tree.cpu(), pred_labels, labels_target.cpu())

    return uas_score, las_score, uas_chuliu

if __name__ == '__main__':
    import pickle
    import dataset, test, model

    dics = pickle.load(open('dictionary.pkl', 'rb'))
    dic_form = dics['dic_form']
    dic_pos = dics['dic_pos']
    dic_label = dics['dic_label']
    embedding = pickle.load(open('embedding.pkl', 'rb'))
    pretrained_word_embeddings, pretrained_pos_embeddings = embedding
    model = model.BiLSTMParser(pretrained_word_embeddings, pretrained_pos_embeddings, 100, 20, n_label=len(dic_label.i2w))
    batch_size = 1
    testset = dataset.Dataset(dic_form, dic_pos, dic_label)
    testset.load('test_devconll06.txt', training=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    if torch.cuda.is_available() and CUDA:
        model.cuda()

    for batch_ndx, sample in enumerate(testset):
        # print(sample)
        score = test.test(model, sample)

        print(score)
        # exit()