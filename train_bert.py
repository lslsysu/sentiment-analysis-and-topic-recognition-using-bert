import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, BertModel
import torch
import torch.nn as nn
import utils
import model

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

# construct vocab
vocab_dict = tokenizer.get_vocab()
vocab = utils.Vocab()
vocab.construct_vocab(vocab_dict)

# construct dataset
train_data = utils.load_data("./data/train.txt")
test_data = utils.load_data("./data/test.txt")

torch.save(train_dataset, './result/train_dataset.pth')
torch.save(test_dataset, './result/test_dataset.pth')

train_dataset = utils.llmstage0Dataset(train_data, max_len = 125, tokenizer = tokenizer, vocab = vocab)
test_dataset = utils.llmstage0Dataset(test_data, max_len = 125, tokenizer = tokenizer, vocab = vocab)

batch_size = 256
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False)

# construct model
num_topic_labels, num_sentiment_labels = 10, 3
net = model.MultiTaskBERT(num_topic_labels, num_sentiment_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
# trainer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
# loss
topic_loss_func = nn.BCEWithLogitsLoss()  
sentiment_loss_func = nn.CrossEntropyLoss() 

# train bert
num_epochs = 50
net, net_loss, topic_metrics, sentiment_metrics = utils.train_bert(net, num_epochs, train_iter, topic_loss_func, sentiment_loss_func, optimizer, device)

torch.save(net.state_dict(), './result/bert_trained.pth')

# visualization
visualize_epochs(net_loss[0], net_loss[1], net_loss[2], 
            topic_metrics[0], topic_metrics[1], topic_metrics[2], topic_metrics[3],
            sentiment_metrics[0], sentiment_metrics[1], sentiment_metrics[2], sentiment_metrics[3])