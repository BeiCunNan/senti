from torch import nn


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
