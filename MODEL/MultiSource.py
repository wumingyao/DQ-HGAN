import torch.nn as nn
from transformers import BertModel


class MultiSourceEncoder(nn.Module):
    def __init__(self, bert_model_path, intent_vocab_size, emotion_vocab_size, hidden_size):
        super(MultiSourceEncoder, self).__init__()

        # Historical dialogue transformer encoder
        self.bert = BertModel.from_pretrained(bert_model_path)

        # Intent extractor
        self.intent_embeddings = nn.Embedding(intent_vocab_size, hidden_size)
        self.intent_encoder = nn.TransformerEncoderLayer(hidden_size, 2)

        # Emotion classifier
        self.emotion_embeddings = nn.Embedding(emotion_vocab_size, hidden_size)
        self.emotion_encoder = nn.TransformerEncoderLayer(hidden_size, 2)
        self.emotion_classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids, intent_ids, emotion_ids):
        # Historical dialogue transformer encoder
        encoded_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[
            0]  # [batch_size, seq_len, hidden_size]

        # Intent extractor
        intent_embeddings = self.intent_embeddings(intent_ids)  # [batch_size, seq_len, hidden_size]
        intent_output = self.intent_encoder(intent_embeddings.transpose(0, 1))  # [seq_len, batch_size, hidden_size]
        intent_feature = intent_output[-1]  # [batch_size, hidden_size]

        # Emotion classifier
        emotion_embeddings = self.emotion_embeddings(emotion_ids)  # [batch_size, seq_len, hidden_size]
        emotion_output = self.emotion_encoder(emotion_embeddings.transpose(0, 1))  # [seq_len, batch_size, hidden_size]
        emotion_feature = self.emotion_classifier(emotion_output[-1]).squeeze()  # [batch_size]
        emotion_feature = self.sigmoid(emotion_feature)  # [batch_size]

        return encoded_output, intent_feature, emotion_feature
