import math

import torch
from torch import nn


class AttentionPooling_a(nn.Module):
    def __init__(self, input_size):
        super(AttentionPooling_a, self).__init__()

        self.fc = nn.Linear(input_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, input_size]

        # 计算注意力权重
        attention_weights = self.fc(inputs)
        attention_weights = self.softmax(attention_weights)

        # 对每个时间步的输出加权求和
        pooled_output = torch.sum(attention_weights * inputs, dim=1)

        return pooled_output


class AttentionPooling_b(nn.Module):

    def __init__(self, input_size):
        super(AttentionPooling_b, self).__init__()

        self.fc = nn.Linear(input_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, input_size]

        # 计算注意力权重
        attention_weights = self.fc(inputs)
        attention_weights = self.softmax(attention_weights)

        # 对每个时间步的输出加权求和
        pooled_output = torch.sum(attention_weights * inputs, dim=1)

        return pooled_output


class A(nn.Module):
    def __init__(self, base_model, num_classes, max_lengths, query_lengths, cls_model, prompt_model, prompt_lengths):
        super().__init__()
        self.base_model = base_model
        self.cls_model = cls_model
        self.prompt_model = prompt_model
        self.num_classes = num_classes
        self.max_lengths = max_lengths
        self.query_lengths = query_lengths + 1
        self.prompt_lengths = prompt_lengths + 1

        for param in base_model.parameters():
            param.requires_grad = (True)

        # Model a
        self.akey_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.aquery_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.avalue_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.a_norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.af_key_layer = nn.Linear(self.max_lengths + self.query_lengths,
                                      self.max_lengths + self.query_lengths)
        self.af_query_layer = nn.Linear(self.max_lengths + self.query_lengths,
                                        self.max_lengths + self.query_lengths)
        self.af_value_layer = nn.Linear(self.max_lengths + self.query_lengths,
                                        self.max_lengths + self.query_lengths)
        self.af_norm_fact = 1 / math.sqrt(self.max_lengths + self.query_lengths)

        # Model b
        self.bkey_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.bquery_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.bvalue_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.b_norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.bf_key_layer = nn.Linear(self.max_lengths, self.max_lengths)
        self.bf_query_layer = nn.Linear(self.max_lengths, self.max_lengths)
        self.bf_value_layer = nn.Linear(self.max_lengths, self.max_lengths)
        self.bf_norm_fact = 1 / math.sqrt(self.max_lengths)

        # Model c
        self.ckey_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.cquery_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.cvalue_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.c_norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.cf_key_layer = nn.Linear(self.max_lengths + self.prompt_lengths,
                                      self.max_lengths + self.prompt_lengths)
        self.cf_query_layer = nn.Linear(self.max_lengths + self.prompt_lengths,
                                        self.max_lengths + self.prompt_lengths)
        self.cf_value_layer = nn.Linear(self.max_lengths + self.prompt_lengths,
                                        self.max_lengths + self.prompt_lengths)
        self.cf_norm_fact = 1 / math.sqrt(self.max_lengths + self.prompt_lengths)

        # self.fnn = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(self.base_model.config.hidden_size * 2, self.base_model.config.hidden_size),
        #     nn.Linear(self.base_model.config.hidden_size, num_classes)
        # )

        self.fnn = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear((1000 + self.base_model.config.hidden_size) * 3, self.base_model.config.hidden_size),
            nn.Linear(self.base_model.config.hidden_size, num_classes)
        )

        self.atW = nn.Linear(self.base_model.config.hidden_size, 100)
        self.afW = nn.Linear(self.base_model.config.hidden_size, 100)
        self.btW = nn.Linear(self.base_model.config.hidden_size, 100)
        self.bfW = nn.Linear(self.base_model.config.hidden_size, 100)
        self.ctW = nn.Linear(self.base_model.config.hidden_size, 100)
        self.cfW = nn.Linear(self.base_model.config.hidden_size, 100)
        self.aftW = nn.Sequential(
            # nn.GELU(),LUA
            nn.Linear(10000, 1000)
        )
        self.bftW = nn.Sequential(
            # nn.GELU(),
            nn.Linear(10000, 1000)
        )
        self.cftW = nn.Sequential(
            # nn.GELU(),
            nn.Linear(10000, 1000)
        )

        self.A_Att_Pooling = AttentionPooling_a(self.base_model.config.hidden_size * 1)
        self.B_Att_Pooling = AttentionPooling_b(self.base_model.config.hidden_size * 1)

    def forward(self, inputs, inputs_cls, inputs_prompt):
        # tokens = self.base_model(**inputs).last_hidden_state
        # cls_tokens = self.cls_model(**inputs_cls).last_hidden_state
        prompt_tokens = self.prompt_model(**inputs_prompt).last_hidden_state

        # CLS = tokens[:, 0, :]
        # cls_CLS = cls_tokens[:, 0, :]
        MASK = prompt_tokens[:, 4, :]

        # tokens_padding = F.pad(tokens[:, 1:, :].permute(0, 2, 1),
        #                        (0, self.max_lengths + self.query_lengths - tokens[:, 1:, :].shape[1]),
        #                        mode='constant',
        #                        value=0).permute(0, 2, 1)
        # cls_padding = F.pad(cls_tokens[:, 1:, :].permute(0, 2, 1),
        #                     (0, self.max_lengths - cls_tokens[:, 1:, :].shape[1]),
        #                     mode='constant',
        #                     value=0).permute(0, 2, 1)

        # prompt_padding = F.pad(torch.cat((prompt_tokens[:, :3, :], prompt_tokens[:, 4:, :]), dim=1).permute(0, 2, 1),
        #                        (0, self.max_lengths + self.prompt_lengths - prompt_tokens[:, 1:, :].shape[1]),
        #                        mode='constant',
        #                        value=0).permute(0, 2, 1)

        # Model a
        # TSA && FSA
        # aK = self.akey_layer(tokens_padding)
        # aQ = self.aquery_layer(tokens_padding)
        # aV = self.avalue_layer(tokens_padding)
        # aattention = nn.Softmax(dim=-1)((torch.bmm(aQ, aK.permute(0, 2, 1))) * self.a_norm_fact)
        # aTSA = torch.bmm(aattention, aV)
        #
        # aK_N = self.af_key_layer(tokens_padding.permute(0, 2, 1))
        # aQ_N = self.af_query_layer(tokens_padding.permute(0, 2, 1))
        # aV_N = self.af_value_layer(tokens_padding.permute(0, 2, 1))
        # aattention_N = nn.Softmax(dim=-1)((torch.bmm(aQ_N, aK_N.permute(0, 2, 1))) * self.af_norm_fact)
        # aFSA = torch.bmm(aattention_N, aV_N).permute(0, 2, 1)

        # Weaver
        # aTSA_W = self.atW(aTSA)
        # aFSA_W = self.afW(aFSA)
        # a_TFSA_W = torch.bmm(aTSA_W.permute(0, 2, 1), aFSA_W)
        # a_TFSA = self.aftW(torch.reshape(a_TFSA_W, [a_TFSA_W.shape[0], 10000]))

        # Model b
        # TSA && FSA
        # bK = self.bkey_layer(cls_padding)
        # bQ = self.bquery_layer(cls_padding)
        # bV = self.bvalue_layer(cls_padding)
        # battention = nn.Softmax(dim=-1)((torch.bmm(bQ, bK.permute(0, 2, 1))) * self.b_norm_fact)
        # bTSA = torch.bmm(battention, bV)
        #
        # bK_N = self.bf_key_layer(cls_padding.permute(0, 2, 1))
        # bQ_N = self.bf_query_layer(cls_padding.permute(0, 2, 1))
        # bV_N = self.bf_value_layer(cls_padding.permute(0, 2, 1))
        # battention_N = nn.Softmax(dim=-1)((torch.bmm(bQ_N, bK_N.permute(0, 2, 1))) * self.bf_norm_fact)
        # bFSA = torch.bmm(battention_N, bV_N).permute(0, 2, 1)

        # Weaver
        # bTSA_W = self.btW(bTSA)
        # bFSA_W = self.bfW(bFSA)
        # b_TFSA_W = torch.bmm(bTSA_W.permute(0, 2, 1), bFSA_W)
        # b_TFSA = self.bftW(torch.reshape(b_TFSA_W, [b_TFSA_W.shape[0], 10000]))

        # model c
        # TSA && FSA
        # cK = self.ckey_layer(prompt_padding)
        # cQ = self.cquery_layer(prompt_padding)
        # cV = self.cvalue_layer(prompt_padding)
        # cattention = nn.Softmax(dim=-1)((torch.bmm(cQ, cK.permute(0, 2, 1))) * self.c_norm_fact)
        # cTSA = torch.bmm(cattention, cV)
        #
        # cK_N = self.cf_key_layer(prompt_padding.permute(0, 2, 1))
        # cQ_N = self.cf_query_layer(prompt_padding.permute(0, 2, 1))
        # cV_N = self.cf_value_layer(prompt_padding.permute(0, 2, 1))
        # cattention_N = nn.Softmax(dim=-1)((torch.bmm(cQ_N, cK_N.permute(0, 2, 1))) * self.cf_norm_fact)
        # cFSA = torch.bmm(cattention_N, cV_N).permute(0, 2, 1)

        # Weaver
        # cTSA_W = self.ctW(cTSA)
        # cFSA_W = self.cfW(cFSA)
        # c_TFSA_W = torch.bmm(cTSA_W.permute(0, 2, 1), cFSA_W)
        # c_TFSA = self.cftW(torch.reshape(c_TFSA_W, [c_TFSA_W.shape[0], 10000]))

        # output_ALL = torch.cat((CLS, cls_CLS, MASK, a_TFSA, b_TFSA, c_TFSA), 1)

        predicts = self.fnn(MASK)

        return predicts
