import torch
import torch.nn as nn

CUDA = True

class BiLSTMParser(nn.Module):
    def __init__(self, pretrained_word_embeddings, pretrained_pos_embeddings,
                 len_word_embed, len_pos_embed, len_feature_vec=20, lstm_hidden_size=400,
                 fc_arc_hidden_size=500, fc_label_hidden_size=200, n_label=47):

        super(BiLSTMParser, self).__init__()
        self.len_word_embed = len_word_embed
        self.len_pos_embed = len_pos_embed
        self.len_data_vec = len_word_embed + len_pos_embed
        self.len_feature_vec = len_feature_vec
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_arc_hidden_size = fc_arc_hidden_size
        self.fc_label_hidden_size = fc_label_hidden_size
        self.n_label = n_label
        self.LSTM_layers = 3

        # trainable parameters
        self.word_embeddings = torch.nn.Embedding(len(pretrained_word_embeddings), len_word_embed)
        self.word_embeddings.weight = torch.nn.Parameter(pretrained_word_embeddings)
        self.pos_embeddings = torch.nn.Embedding(len(pretrained_pos_embeddings), len_pos_embed)
        self.pos_embeddings.weight = torch.nn.Parameter(pretrained_pos_embeddings)

        self.BiLSTM = torch.nn.LSTM(input_size=self.len_data_vec, hidden_size=self.lstm_hidden_size,
                                    num_layers = self.LSTM_layers, dropout=.33, bidirectional=True)

        self.FC_arc_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, fc_arc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_arc_hidden_size, len_feature_vec)
        )
        self.FC_arc_dep = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, fc_arc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_arc_hidden_size, len_feature_vec)
        )
        self.FC_label_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, fc_arc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_arc_hidden_size, len_feature_vec)
        )
        self.FC_label_dep = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2, fc_arc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_arc_hidden_size, len_feature_vec)
        )
        self.FC_label_classifier = nn.Sequential(
            nn.Linear(self.len_feature_vec*2, self.len_feature_vec),
            nn.ReLU(),
            nn.Linear(self.len_feature_vec, self.n_label)
        )

        self.U_1 = nn.Parameter(torch.randn(len_feature_vec, len_feature_vec))
        self.u_2 = nn.Parameter(torch.randn(1, len_feature_vec))

    def forward(self, sequence):
        seq_len = len(sequence[0])
        word_sequence = sequence[:,0]
        pos_sequence = sequence[:,1]
        gold_tree = sequence[:,2] if seq_len == 3 else None # if there is no gold tree given, only predict arcs, not labels

        word_embeddings = self.word_embeddings(word_sequence)
        pos_embeddings = self.pos_embeddings(pos_sequence)
        x = torch.cat((word_embeddings, pos_embeddings), 1)
        x = x[:, None, :]  # add a batch-dimension

        # initialise hidden state of the LSTM
        h = torch.zeros(self.LSTM_layers * 2, 1, self.lstm_hidden_size)
        c = torch.zeros(self.LSTM_layers * 2, 1, self.lstm_hidden_size) # batch_size fix to 1
        if torch.cuda.is_available() and CUDA:
            h = h.cuda()
            c = c.cuda()
        hidden = (h, c)

        r, _ = self.BiLSTM(x, hidden)

        #arcs
        h_arc_head = torch.squeeze(self.FC_arc_head(r))
        h_arc_dep = torch.squeeze(self.FC_arc_dep(r))
        adj_matrix = h_arc_head @ self.U_1 @ torch.t(h_arc_dep) + h_arc_head @ torch.t(self.u_2)

        # if we only try to predict arcs (not labels) we will skip this block
        # this occurs during testing
        pred_labels = None
        if gold_tree is not None:
            h_label_head = torch.squeeze(self.FC_label_head(r))
            h_label_dep = torch.squeeze(self.FC_label_dep(r))
            h_label_dep = h_label_dep[gold_tree.data]
            arcs_to_label = torch.cat((h_label_head, h_label_dep),1)
            pred_labels = self.FC_label_classifier(arcs_to_label)

        return adj_matrix, pred_labels