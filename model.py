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
        print(cls_feats.size())
        predicts = self.block(cls_feats)
        return predicts


# One method
#         input_ids1 = inputs['input_ids']
#         token_type_ids1 = inputs['token_type_ids'][0]
#         attention_mask1 = inputs['attention_mask']
#         raw_outputs = self.base_model(input_ids=input_ids1,
#                                       token_type_ids=token_type_ids1,
#                                       attention_mask=attention_mask1)


class Capsule_Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        label_feats = raw_outputs.last_hidden_state[:, 1:self.num_classes + 1, :]
