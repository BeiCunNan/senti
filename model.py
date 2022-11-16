import torch
from torch import nn
import torch.nn.functional as F


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
        self.num_classes = num_classes
        self.base_model = base_model
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 2
        self.encode_layer = 12
        self.num_filter_total = self.num_filters * len(self.filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, self.num_classes, bias=False)
        self.bias = nn.Parameter(torch.ones([self.num_classes]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, kernel_size=(size, self.base_model.config.hidden_size)) for size in
            self.filter_sizes
        ])

    def forward(self, x):
        # x: [batch_size, 12, hidden]
        x = x.unsqueeze(1)  # [batch_size, channel=1, 12, hidden]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            out = F.relu(conv(x))  # [batch_size, channel=2, 12-kernel_size[0]+1, 1]
            maxPool = nn.MaxPool2d(
                kernel_size=(self.encode_layer - self.filter_sizes[i] + 1, 1)
            )
            # maxPool: [batch_size, channel=2, 1, 1]
            pooled = maxPool(out).permute(0, 3, 2, 1)  # [batch_size, h=1, w=1, channel=2]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes))  # [batch_size, h=1, w=1, channel=2 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])  # [batch_size, 6]

        output = self.Weight(h_pool_flat) + self.bias  # [batch_size, class_num]

        return output


class Transformer_Text_Last_Hidden(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 2
        self.encode_layer = 12

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(k, self.base_model.config.hidden_size), ) for k in self.filter_sizes]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)

    def conv_pool(self, x, conv):
        x = conv(x)  # shape [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1, 1]
        x = F.relu(x)
        x = x.squeeze(3)  # shape [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1]
        size = x.size(2)
        x = F.max_pool1d(x, size)  # shape[batch+size, out_channels, 1]
        x = x.squeeze(2)  # shape[batch+size, out_channels]
        return x

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        out = raw_outputs.last_hidden_state.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        out = torch.cat([self.conv_pool(out, conv) for conv in self.convs],
                        1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]
        out = self.dropout(out)
        out = self.fc(out)
        return out


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
        self.linear = nn.Linear(self.base_model.config.hidden_size, self.num_classes)
        self.textCnn = TextCnnModel(self.base_model, self.num_classes)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs, output_hidden_states=True)
        # 取每一层encode出来的向量
        hidden_states = raw_outputs.hidden_states  # 13 * [batch_size, seq_len, hidden] 第一层是 embedding 层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [batch_size, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textCnn的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [batch_size, 12, hidden]
        pred = self.textCnn(cls_embeddings)
        return pred
