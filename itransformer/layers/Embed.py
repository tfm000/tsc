import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # transposing x for multivariate attention
        x = x.permute(0, 2, 1)

        # embedding
        x = self.value_embedding(x)
        return self.dropout(x)

        # # x: [Batch Variate Time]
        # if x_mark is None:
        #     x = self.value_embedding(x)
        # else:
        #     # the potential to take covariates (e.g. timestamps) as tokens
        #     x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        #     ######### Tyler Comment
        #     # Here he is transposing the position embeddings (as iTransformer 
        #     # has data transposed for the attention module) and the combining 
        #     # with the rest of the data. Seems very wasteful to do this inside 
        #     # of the model, when this is really just data preprocessing
        #     #########
            
        # # x: [Batch Variate d_model]
        # return self.dropout(x)

