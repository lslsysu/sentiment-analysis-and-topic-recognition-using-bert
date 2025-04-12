import torch.nn as nn
from transformers import BertModel

class MultiTaskBERT(nn.Module):
    def __init__(self, num_topic_labels, num_sentiment_labels, dropout = 0.1):
        super(MultiTaskBERT, self).__init__()
        # pre-trained Bert
        self.bert = BertModel.from_pretrained("google-bert/bert-base-chinese")
        self.dropout = nn.Dropout(dropout)
        # 两个任务的神经网络层
        self.topic_classifier = nn.Linear(self.bert.config.hidden_size, num_topic_labels)
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, num_sentiment_labels)
        # self.relu = nn.ReLU()
        
    def forward(self, token_ids, attention_mask):
        # 获取BERT的输出
        outputs = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        # 获取最后一层隐藏状态（last_hidden_state），并选择[CLS] token的表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        # 暂退层
        dropout_output = self.dropout(cls_output)
        # 任务特定的输出层
        topic_output = self.topic_classifier(dropout_output)       
        sentiment_output = self.sentiment_classifier(dropout_output)
        
        return topic_output, sentiment_output