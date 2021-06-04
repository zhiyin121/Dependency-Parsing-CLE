import torch
import dataset, test, model
import os, pickle
import json


# load data
if not os.path.exists('dictionary.de.pkl'):
    dic_form = dataset.Dictionary()
    dic_pos = dataset.Dictionary()
    dic_label = dataset.Dictionary()
else:
    dics = pickle.load(open('dictionary.de.pkl', 'rb'))
    dic_form = dics['dic_form']
    dic_pos = dics['dic_pos']
    dic_label = dics['dic_label']

trainset = dataset.Dataset(dic_form, dic_pos, dic_label)
trainset.load('german/train/tiger-2.2.train.conll06')

testset = dataset.Dataset(dic_form, dic_pos, dic_label)
testset.load('german/dev/tiger-2.2.dev.conll06.gold', training=False)

pickle.dump({
    'dic_form': dic_form,
    'dic_pos': dic_pos,
    'dic_label': dic_label
}, open('dictionary.de.pkl', 'wb'))


batch_size = 1
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# load embedding
if not os.path.exists('embedding.de.pkl'):
    pretrained_word_embeddings, pretrained_pos_embeddings = dataset.pretrain_word_embeddings(trainset, 100, 20, dic_form, dic_pos)
else:
    embedding = pickle.load(open('embedding.de.pkl', 'rb'))
    pretrained_word_embeddings, pretrained_pos_embeddings = embedding

pickle.dump([
    pretrained_word_embeddings,
    pretrained_pos_embeddings
], open('embedding.de.pkl', 'wb'))


# init model
model = model.BiLSTMParser(pretrained_word_embeddings, pretrained_pos_embeddings, 100, 20, n_label=len(dic_label.i2w))
criterion = torch.nn.CrossEntropyLoss()
lr = 0.002
weight_decay = 1e-6
betas = (0.9, 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

if torch.cuda.is_available():
    model.cuda()

record = []
score_record = []
best = 0
for epoch in range(10):
    # start training
    model.train()
    loss_record = 0.0
    for batch_ndx, sample in enumerate(train_loader):
        seq_len = len(sample['form'])
        if seq_len <= 1: continue

        gold_tree = torch.LongTensor(sample['tree'])
        labels_target = torch.LongTensor(sample['label'])
        
        sequence = torch.LongTensor(seq_len, 3)
        sequence[:,2] = gold_tree
        sequence[:,1] = torch.LongTensor(sample['pos'])
        sequence[:,0] = torch.LongTensor(sample['form'])

        # Move to GPU
        if torch.cuda.is_available():
            gold_tree = gold_tree.cuda()
            labels_target = labels_target.cuda()
            sequence = sequence.cuda()

        # forward
        optimizer.zero_grad()
        adj_mat, labels_pred = model(sequence)

        # measuring losses
        arc_pred = torch.t(adj_mat)  # nn.CrossEntropyLoss() wants the classes in the second dimension

        arc_loss = criterion(arc_pred, gold_tree)
        _, predicted = torch.max(arc_pred, 1)
        arc_acc = (predicted == gold_tree).sum().item()/seq_len
        label_loss = criterion(labels_pred, labels_target)
        _, predicted = torch.max(labels_pred, 1)
        lab_acc = (predicted == labels_target).sum().item()/seq_len
        total_loss = arc_loss + label_loss

        # backprop
        total_loss.backward()
        optimizer.step()

        loss_record += total_loss.item()
        if batch_ndx % 10 == 0:
            print('\repoch:', epoch, 'i:', batch_ndx, 'loss:', round(loss_record / 500, 4), 'arc_acc:', round(arc_acc, 2),'lab_acc:', round(lab_acc, 2), end='')
            record.append({
                'epoch': epoch,
                'batch_ndx': batch_ndx,
                'loss': loss_record / 10
            })
            loss_record = 0.0
            json.dump(record, open('de_record.json', 'w'), indent=2)
            # break
            
    # evaluation

    model.eval()
    uas_chuliu = 0
    uas_scores = 0
    las_scores = 0
    arc_count = 0
    for batch_ndx, sample in enumerate(test_loader):
        score = test.test(model, sample)
        #print(score)
        if score is None:
            pass
        else:
            uas, las, chuliu = score
            uas_chuliu += chuliu[0]
            uas_scores += uas[0]
            las_scores += las[0]
            arc_count += uas[1]
    print('\nTest epoch:', epoch, 'UAS:', round((uas_scores/arc_count)*100, 4), 'LAS:', round((las_scores/arc_count)*100, 4), 'Chu-Liu:', round((uas_chuliu/arc_count)*100, 4))
    score_record.append({
        'epoch': epoch,
        'UAS': (uas_scores/arc_count)*100, 
        'LAS': (las_scores/arc_count)*100,
        'Chu-Liu': (uas_chuliu/arc_count)*100
    })
    if uas_scores > best:
        torch.save(model, open('model.de.pt', 'wb'))
        best = uas_scores
    json.dump(score_record, open('de_record_score.json', 'w'), indent=2)
            