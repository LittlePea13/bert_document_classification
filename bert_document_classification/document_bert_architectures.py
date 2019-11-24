from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from torch import nn
import torch
from .transformer import TransformerEncoderLayer, TransformerEncoder

from torch.nn import LSTM
class DocumentBertLSTM(BertPreTrainedModel):
    """
    BERT output over document in LSTM
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, freeze_bert=True, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(freeze_bert):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

        output, (_, _) = self.lstm(bert_output.permute(1,0,2))
        del(bert_output)
        #print("Last LSTM layer shape:",last_layer.shape)
        prediction = self.classifier(output[-1])
        del(output)
        #print("Prediction Shape", prediction.shape)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class DocumentBertLinear(BertPreTrainedModel):
    """
    BERT output over document into linear layer
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertLinear, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        #self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(bert_model_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * self.bert_batch_size, bert_model_config.num_labels),
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, freeze_bert=True, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        del(document_batch)
        use_grad = not freeze_bert
        with torch.set_grad_enabled(use_grad):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])


        prediction = self.classifier(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class DocumentBertMaxPool(BertPreTrainedModel):
    """
    BERT output over document into linear layer
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertMaxPool, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        # self.transformer_encoder = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
        #                                            nhead=6,
        #                                            dropout=bert_model_config.hidden_dropout_prob)
        #self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(bert_model_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, freeze_bert=True, device='cuda'):


        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(use_grad):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])


        prediction = self.classifier(bert_output.max(dim=1)[0])
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

class DocumentBertMean(BertPreTrainedModel):
    """
    BERT output over document into an averaged hidden state and then a linear layer
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertMean, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        # self.transformer_encoder = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
        #                                            nhead=6,
        #                                            dropout=bert_model_config.hidden_dropout_prob)
        #self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(bert_model_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, freeze_bert=True, device='cuda'):


        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(use_grad):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])


        prediction = self.classifier(bert_output.mean(dim=0))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class DocumentBertTransformer(BertPreTrainedModel):
    """
    BERT -> TransformerEncoder -> Max over attention output.
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertTransformer, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        encoder_layer = TransformerEncoderLayer(d_model=bert_model_config.hidden_size,
                                                   nhead=6,
                                                   dropout=bert_model_config.hidden_dropout_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, freeze_bert=True, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device='cuda')

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(use_grad):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

        transformer_output = self.transformer_encoder(bert_output.permute(1,0,2))

        #print(transformer_output.shape)

        prediction = self.classifier(transformer_output.permute(1,0,2).max(dim=1)[0])
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

class DocumentBertLSTMAtt(BertPreTrainedModel):
    """
    BERT output over document in LSTM
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertLSTMAtt, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )
        self.attention = AttentionModule(bert_model_config.hidden_size,
            batch_first=True,
            layers=1,
            dropout=.0,
            non_linearity="tanh")

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, freeze_bert=True, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(use_grad):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])
        output, (_, _) = self.lstm(bert_output.permute(1,0,2))
        del(bert_output)
        #last_layer = output[-1]
        #print("Last LSTM layer shape:",last_layer.shape)
        attention_output, _, _ = self.attention.forward(inputs = output.permute(1,0,2))
        del(output)
        prediction = self.classifier(attention_output)
        #print("Prediction Shape", prediction.shape)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

class DocumentBertAtt(BertPreTrainedModel):
    """
    BERT output over document in Attention
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertAtt, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels),
        )
        self.attention = AttentionModule(bert_model_config.hidden_size,
            batch_first=True,
            layers=1,
            dropout=.0,
            non_linearity="tanh")

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, freeze_bert=True, device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        use_grad = not freeze_bert
        with torch.set_grad_enabled(use_grad):
            for doc_id in range(document_batch.shape[0]):
                bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])
        #last_layer = output[-1]
        #print("Last LSTM layer shape:",last_layer.shape)
        attention_output, _, _ = self.attention.forward(inputs = bert_output)
        del(bert_output)
        prediction = self.classifier(attention_output)
        #print("Prediction Shape", prediction.shape)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

class AttentionModule(nn.Module):
    def __init__(
        self,
        attention_size,
        batch_first=True,
        layers=1,
        dropout=.0,
        non_linearity="tanh"):
        super(AttentionModule, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for _ in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*modules)
        modules_att = []
        # last attention layer must output 1
        modules_att.append(nn.Linear(attention_size, 1))
        modules_att.append(activation)
        modules_att.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules_att)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        linear_output = self.linear_layer(inputs)
        scores = self.attention(linear_output).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(linear_output, scores.unsqueeze(-1).expand_as(linear_output))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores, weighted