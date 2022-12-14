import math

import torch
from torch import nn
import torch.nn.functional as F

from pooling import MaxPooling, MeanPooling, WeightedLayerPooling, AttentionPooling, Max_KMeanPooling


class Transformer_CLS(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_model.config.hidden_size, 192),
            nn.Linear(192, 24),
            nn.Linear(24, num_classes),
            nn.Softmax(dim=1)
        )
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state[:, 0, :]
        predicts = self.block(cls_feats)
        return predicts


class Transformer_Extend_LSTM(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        lstm_outputs, _ = self.lstm(tokens)

        lstm_outputs = lstm_outputs[:, -1, :]
        predicts = self.block(lstm_outputs)
        return predicts


class Transformer_Extend_BILSTM(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        lstm_outputs, _ = self.lstm(tokens)

        lstm_outputs = lstm_outputs[:, -1, :]
        predicts = self.block(lstm_outputs)
        return predicts


class Capsule_Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        label_feats = raw_outputs.last_hidden_state[:, 1:self.num_classes + 1, :]


class TextCnnModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(TextCnnModel, self).__init__()

        # Define the hyperparameters
        self.num_classes = num_classes
        self.base_model = base_model
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 2
        self.encode_layer = 12

        # Define some network parameters
        self.num_filter_total = self.num_filters * len(self.filter_sizes)  # ??????6???filters
        self.Weight = nn.Linear(self.num_filter_total, self.num_classes, bias=False)  # ???????????????
        self.bias = nn.Parameter(torch.ones([self.num_classes]))  # tensor([1.,1.]) ???????????????bias

        # ?????????inchannels=1,outchannels=2,????????????[2,768]???[3,768]???[4,768]?????????
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, kernel_size=(size, self.base_model.config.hidden_size)) for size in
            self.filter_sizes
        ])

    def forward(self, x):
        # x: [batch_size, 12, hidden]
        x = x.unsqueeze(1)  # [batch_size, channel=1, 12, hidden] -> [8,1,12,768]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            out = F.relu(
                conv(x))  # [batch_size, channel=2, 12-kernel_size[0]+1, 1]  ->[8, 2, 11, 1]???[8, 2, 10, 1]???[8, 2, 9, 1]
            maxPool = nn.MaxPool2d(
                kernel_size=(self.encode_layer - self.filter_sizes[i] + 1, 1)
            )
            # maxPool(out): [batch_size, channel=2, 1, 1] -> [8, 2, 1, 1]
            # pooled : [8, 1, 1, 2]
            pooled = maxPool(out).permute(0, 3, 2, 1)  # [batch_size, h=1, w=1, channel=2]
            pooled_outputs.append(pooled)

        # ?????????3??????Tensor????????????
        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes))  # [batch_size, h=1, w=1, channel=2 * 3]

        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])  # [batch_size, 6]

        # output = self.Weight(h_pool_flat) + self.bias  # [batch_size, class_num]
        output = h_pool_flat

        return output


class Transformer_Text_Last_Hidden(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        out = torch.cat([self.conv_pool(tokens, conv) for conv in self.convs],
                        1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]
        predicts = self.block(out)
        return predicts


class Transformer_Text_Hiddens(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 2
        self.encode_layer = 12
        self.textCnn = TextCnnModel(self.base_model, self.num_classes)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs, output_hidden_states=True)
        # ????????????encode???????????????
        hidden_states = raw_outputs.hidden_states  # 13 * [batch_size, seq_len, hidden] ???????????? embedding ????????????
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [batch_size, 1, hidden]
        # ????????????????????????token(cls??????)?????????????????????????????????textCnn?????????
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [batch_size, 12, hidden]
        pred = self.textCnn(cls_embeddings)
        pred = self.block(pred)
        return pred


class Transformer_CNN_RNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cnn_tokens = raw_outputs.last_hidden_state.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]
        rnn_tokens = raw_outputs.last_hidden_state
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch,300]
        # rnn_out --> [batch,512]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)
        return predicts


class Gate_Residual_CNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        # self.conv1d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, self.base_model.config.hidden_size),dilation=(,1))

        for param in base_model.parameters():
            param.requires_grad = (True)

    def conf_pool(self):
        pass

    def gate(self):
        pass

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)


class ExplainableModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        for param in base_model.parameters():
            param.requires_grad = (True)

        self.span_info_collect = SICModel(self.base_model.config.hidden_size)
        self.interpretation = InterpretationModel(self.base_model.config.hidden_size)
        self.output = nn.Linear(self.base_model.config.hidden_size, self.num_classes)

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = inputs['position_ids']
        start_indexs = inputs['start_indexs']
        end_indexs = inputs['end_indexs']
        span_masks = inputs['span_masks']

        # intermediate layer
        raw_outputs = self.base_model(input_ids,
                                      attention_mask=attention_mask)  # output.shape = (bs, length, hidden_size)
        hidden_states = raw_outputs.last_hidden_state
        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(hidden_states, start_indexs, end_indexs)
        # interpretation layer
        H, a_ij = self.interpretation(h_ij, span_masks)
        # output layer
        out = self.output(H.float())
        return out, a_ij


class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, start_indexs, end_indexs):
        W1_h = self.W_1(hidden_states)  # (bs, length, hidden_size)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb = torch.index_select(W1_h, 1, start_indexs)  # (bs, span_num, hidden_size)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indexs)
        W3_hi_start_emb = torch.index_select(W3_h, 1, start_indexs)
        W3_hi_end_emb = torch.index_select(W3_h, 1, end_indexs)
        W4_hj_start_emb = torch.index_select(W4_h, 1, start_indexs)
        W4_hj_end_emb = torch.index_select(W4_h, 1, end_indexs)

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi???hj)]
        span = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        h_ij = torch.tanh(span)
        return h_ij


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h_t = nn.Linear(hidden_size, 1)

    def forward(self, h_ij, span_masks):
        # Span mask ??????????????????????????????????????????,?????????[CLS]???[SEP]?????????????????????????????????
        # h_ij --> []
        o_ij = self.h_t(h_ij).squeeze(-1)  # (ba, span_num) [8,406]
        print(1, o_ij[0][:50])
        # print(1, o_ij)
        # print(2, o_ij.shape)
        # mask illegal span
        # print(7,span_masks[0][:50])
        # print(8,span_masks.shape)
        # span_mask -->  []
        o_ij = o_ij - span_masks  # [8,406]
        print(2, o_ij[0][:50])
        # print(4, o_ij.shape)
        # normalize all a_ij, a_ij sum = 1
        a_ij = nn.functional.softmax(o_ij, dim=1)
        # print(6, a_ij.shape) # [8,406]
        print(3, a_ij[0][:50])
        # weight average span representation to get H
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1)  # (bs, hidden_size) [8,768]
        # print(H)
        # print(H.shape)
        return H, a_ij


class Self_Attention(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        for param in base_model.parameters():
            param.requires_grad = (True)

        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)
        self.fnn = nn.Linear(self.base_model.config.hidden_size, num_classes)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        output = torch.bmm(attention, V)
        output = torch.sum(output, dim=1)
        predicts = self.fnn(output)
        return predicts


class Self_Attention_Loss(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        for param in base_model.parameters():
            param.requires_grad = (True)

        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)
        self.fnn = nn.Linear(self.base_model.config.hidden_size, num_classes)

        self.maxpooling = MaxPooling()

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        attention_mask = inputs['attention_mask']
        # print(1,inputs['attention_mask'].unsqueeze(-1).shape)
        # print(2,inputs['attention_mask'].unsqueeze(-1).expand(tokens.size()).float().shape)
        # print(3,inputs['attention_mask'].unsqueeze(-1).expand(tokens.size()).float())
        # print(1,tokens.shape)
        # print(tokens[0][0])

        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q.permute(0, 2, 1), K)) * self._norm_fact)

        all_ij = attention.view(tokens.shape[0], 1, -1)
        all_ij = torch.squeeze(all_ij, 1)

        output = torch.bmm(attention, V.permute(0, 2, 1))
        output = torch.sum(output, dim=2)
        predicts = self.fnn(output)

        return predicts, all_ij


class Self_Attention_New(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        for param in base_model.parameters():
            param.requires_grad = (True)

        self.nsakey_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.nsaquery_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.nsavalue_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.nsa_norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.fnn = nn.Linear(self.base_model.config.hidden_size * 2, num_classes)
        self.sgsa = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        # SA
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self.nsa_norm_fact)
        output = torch.bmm(attention, V)

        # Layer_Normalizaton
        norm = nn.LayerNorm([output.shape[1], output.shape[2]], eps=1e-05).cuda()
        output_LN = norm(output)

        # NSA
        K_N = self.key_layer(output_LN)
        Q_N = self.query_layer(output_LN)
        V_N = self.value_layer(output_LN)
        attention_N = nn.Softmax(dim=-1)((torch.bmm(Q_N.permute(0, 2, 1), K_N) * self._norm_fact))
        output_N = torch.bmm(V_N, attention_N)

        # Add
        # output_N = torch.cat((tokens,output_SGSA), 2)
        output_N = torch.add(tokens,output_N)

        # Layer_Normalization
        norm = nn.LayerNorm([output_N.shape[1], output_N.shape[2]], eps=1e-05).cuda()
        output_LN = norm(output_N)

        # SGSA
        output_SGSA=self.sgsa(output_LN)*output_LN

        # Pooling
        output_A = torch.mean(output_SGSA, dim=1)
        output_B, _ = torch.max(output_SGSA, dim=1)

        predicts = self.fnn(torch.cat((output_A, output_B), 1))
        return predicts
