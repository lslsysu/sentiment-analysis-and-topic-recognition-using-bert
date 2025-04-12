import torch
import utils
import model

# load dataset
test_dataset = torch.load('./result/test_dataset.pth')
batch_size = 256
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

# load model
num_topic_labels, num_sentiment_labels = 10, 3
net = model.MultiTaskBERT(num_topic_labels, num_sentiment_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net.load_state_dict(torch.load('./result/bert_trained.pth', map_location=device))

# test bert
utils.test_bert(net, test_iter, device)