import math
import torch
from torch import nn
import torch.nn.functional as F

class MP_TFWA(nn.Module):
    def __init__(self, base_model,cls_model, prompt_model, num_classes, max_lengths, query_lengths,  prompt_lengths):
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

        # MRC-IE
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

        # Context-IE
        self.bkey_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.bquery_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.bvalue_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.b_norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.bf_key_layer = nn.Linear(self.max_lengths, self.max_lengths)
        self.bf_query_layer = nn.Linear(self.max_lengths, self.max_lengths)
        self.bf_value_layer = nn.Linear(self.max_lengths, self.max_lengths)
        self.bf_norm_fact = 1 / math.sqrt(self.max_lengths)

        # PL-IE
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

        self.fnn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear((1000 + self.base_model.config.hidden_size) * 3, self.base_model.config.hidden_size),
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

    def forward(self, mrc_inputs, text_inputs, mask_inputs, mask_index):
        mrc_tokens = self.base_model(**mrc_inputs).last_hidden_state
        context_tokens = self.cls_model(**text_inputs).last_hidden_state
        pl_tokens = self.prompt_model(**mask_inputs).last_hidden_state

        mrc_CLS = mrc_tokens[:, 0, :]
        context_CLS = context_tokens[:, 0, :]
        MASK = pl_tokens[0, mask_index[0, 1], :].reshape((1, 768))
        for i in range(1, mask_index.shape[0]):
            MASK = torch.cat((MASK, pl_tokens[i, mask_index[i, 1], :].reshape((1, 768))), 0)

        tokens_padding = F.pad(mrc_tokens[:, 1:, :].permute(0, 2, 1),
                               (0, self.max_lengths + self.query_lengths - mrc_tokens[:, 1:, :].shape[1]),
                               mode='constant',
                               value=0).permute(0, 2, 1)
        cls_padding = F.pad(context_tokens[:, 1:, :].permute(0, 2, 1),
                            (0, self.max_lengths - context_tokens[:, 1:, :].shape[1]),
                            mode='constant',
                            value=0).permute(0, 2, 1)
        prompt_padding = F.pad(pl_tokens[:, 1:, :].permute(0, 2, 1),
                               (0, self.max_lengths + self.prompt_lengths - pl_tokens[:, 1:, :].shape[1]),
                               mode='constant',
                               value=0).permute(0, 2, 1)
        # MRC-IE
        aK = self.akey_layer(tokens_padding)
        aQ = self.aquery_layer(tokens_padding)
        aV = self.avalue_layer(tokens_padding)
        aattention = nn.Softmax(dim=-1)((torch.bmm(aQ, aK.permute(0, 2, 1))) * self.a_norm_fact)
        aTSA = torch.bmm(aattention, aV)

        aK_N = self.af_key_layer(tokens_padding.permute(0, 2, 1))
        aQ_N = self.af_query_layer(tokens_padding.permute(0, 2, 1))
        aV_N = self.af_value_layer(tokens_padding.permute(0, 2, 1))
        aattention_N = nn.Softmax(dim=-1)((torch.bmm(aQ_N, aK_N.permute(0, 2, 1))) * self.af_norm_fact)
        aFSA = torch.bmm(aattention_N, aV_N).permute(0, 2, 1)

        aTSA_W = self.atW(aTSA)
        aFSA_W = self.afW(aFSA)
        a_TFSA_W = torch.bmm(aTSA_W.permute(0, 2, 1), aFSA_W)
        a_TFSA = self.aftW(torch.reshape(a_TFSA_W, [a_TFSA_W.shape[0], 10000]))

        # Context-IE
        bK = self.bkey_layer(cls_padding)
        bQ = self.bquery_layer(cls_padding)
        bV = self.bvalue_layer(cls_padding)
        battention = nn.Softmax(dim=-1)((torch.bmm(bQ, bK.permute(0, 2, 1))) * self.b_norm_fact)
        bTSA = torch.bmm(battention, bV)

        bK_N = self.bf_key_layer(cls_padding.permute(0, 2, 1))
        bQ_N = self.bf_query_layer(cls_padding.permute(0, 2, 1))
        bV_N = self.bf_value_layer(cls_padding.permute(0, 2, 1))
        battention_N = nn.Softmax(dim=-1)((torch.bmm(bQ_N, bK_N.permute(0, 2, 1))) * self.bf_norm_fact)
        bFSA = torch.bmm(battention_N, bV_N).permute(0, 2, 1)

        bTSA_W = self.btW(bTSA)
        bFSA_W = self.bfW(bFSA)
        b_TFSA_W = torch.bmm(bTSA_W.permute(0, 2, 1), bFSA_W)
        b_TFSA = self.bftW(torch.reshape(b_TFSA_W, [b_TFSA_W.shape[0], 10000]))

        # PL-IE
        cK = self.ckey_layer(prompt_padding)
        cQ = self.cquery_layer(prompt_padding)
        cV = self.cvalue_layer(prompt_padding)
        cattention = nn.Softmax(dim=-1)((torch.bmm(cQ, cK.permute(0, 2, 1))) * self.c_norm_fact)
        cTSA = torch.bmm(cattention, cV)

        cK_N = self.cf_key_layer(prompt_padding.permute(0, 2, 1))
        cQ_N = self.cf_query_layer(prompt_padding.permute(0, 2, 1))
        cV_N = self.cf_value_layer(prompt_padding.permute(0, 2, 1))
        cattention_N = nn.Softmax(dim=-1)((torch.bmm(cQ_N, cK_N.permute(0, 2, 1))) * self.cf_norm_fact)
        cFSA = torch.bmm(cattention_N, cV_N).permute(0, 2, 1)

        cTSA_W = self.ctW(cTSA)
        cFSA_W = self.cfW(cFSA)
        c_TFSA_W = torch.bmm(cTSA_W.permute(0, 2, 1), cFSA_W)
        c_TFSA = self.cftW(torch.reshape(c_TFSA_W, [c_TFSA_W.shape[0], 10000]))

        output_ALL = torch.cat((mrc_CLS, context_CLS, MASK, a_TFSA, b_TFSA, c_TFSA), 1)
        #output_ALL = torch.cat(( context_CLS ,b_TFSA), 1)

        predicts = self.fnn(output_ALL)

        return predicts,aTSA, aFSA, mrc_tokens, mrc_CLS, bTSA, bFSA, context_tokens, context_CLS,cTSA,cFSA,pl_tokens,MASK
