import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch.optim as optim
import torch.nn as nn
import numpy as np

class Vocab:
    """Vocab class"""
    def __init__(self):
        self.idx_to_token = []
        self.token_to_idx = {}
                
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens)
        return [self.__getitem__(token) for token in tokens]
    
    def construct_vocab(self, vocab_dict):
        self.idx_to_token = [token for token, _ in sorted(vocab_dict.items(), key=lambda x: x[1])]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        return self
    

def load_data(file_path):
    """把text, topic label和sentiment label转化成列表"""
    texts = []
    topic_labels = []
    sentiment_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            text = parts[0].strip().replace(" ", "") 
            labels = parts[1:]
            topic_label = []
            sentiment_label = None
            for label in labels:
                if "#" in label:
                    topic, sentiment = label.split("#")
                    topic_label.append(topic)
                    if sentiment_label is None:
                        sentiment_label = int(sentiment)
            texts.append(text)
            topic_labels.append(topic_label)
            sentiment_labels.append(sentiment_label)
            
            topic_labels_multihot = encode_labels_multi_hot(topic_labels)
            sentiment_labels = [2 if label == -1 else label for label in sentiment_labels]

    return (texts, topic_labels_multihot, sentiment_labels)

def encode_labels_multi_hot(topic_label):
    """将多标签数据转换为multi-hot编码张量"""
    all_labels = sorted(set(l for sample in topic_label for l in sample))
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    
    multi_hot_vectors = []
    for labels in topic_label:
        vec = torch.zeros(len(label2id), dtype=torch.float)
        for label in labels:
            idx = label2id[label]
            vec[idx] = 1.0
        multi_hot_vectors.append(vec)

    topic_label_multihot = torch.stack(multi_hot_vectors)
    return topic_label_multihot

class llmstage0Dataset(torch.utils.data.Dataset):
    """构建Dataset"""
    def __init__(self, data, max_len, tokenizer, vocab):
        # read input data
        all_texts = data[0]
        self.max_len = max_len
        self.all_topic_labels = torch.tensor(data[1], dtype=torch.long)
        self.all_sentiment_labels = torch.tensor(data[2], dtype=torch.long)
        self.vocab = vocab
        
        # tokenize
        all_tokens = [tokenizer.tokenize(text) for text in all_texts]
        
        # construct features 
        (self.all_token_ids, self.all_attention_mask) = zip(*[self.data_preprocess(tokens, max_len, vocab) for tokens in all_tokens])
        
    def data_preprocess(self, tokens, max_len, vocab):
        """Tokenize + Truncate and Padding + Text to Token IDs"""
        # truncate
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]
        # add special token
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        valid_len = len(tokens)
        attention_mask = torch.zeros(max_len, dtype=torch.long)
        attention_mask[:valid_len] = 1
        # Text to Token IDs + Padding
        token_ids = vocab[tokens] + [vocab['[PAD]']] * (max_len - len(tokens))
        
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        valid_len = torch.tensor(valid_len, dtype=torch.long)
        
        return token_ids, attention_mask
        
        
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_attention_mask[idx]), self.all_topic_labels[idx], self.all_sentiment_labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
    
class calc_metrics():
    """calculate evaluation metrics"""
    def __init__(self, true_label, pred_label, class_num, isMultiLabel, device):
        self.true_label = true_label
        self.pred_label = pred_label
        self.class_num = class_num
        self.isMultiLabel = isMultiLabel
        self.device = device
        
    def compute_metrics(self):
        self.accuracy_score = self.accuracy_score(self.true_label, self.pred_label, self.isMultiLabel)
        self.precision_score = self.precision_score(self.true_label, self.pred_label, self.class_num, self.isMultiLabel)
        self.recall_score = self.recall_score(self.true_label, self.pred_label, self.class_num, self.isMultiLabel)
        self.f1_score = self.f1_score(self.true_label, self.pred_label, self.class_num, self.isMultiLabel)
       
    # accuracy score
    def accuracy_score(self, true_label, pred_label, isMultiLabel):
        if isMultiLabel == False:
            return (true_label == pred_label).sum().item() / len(true_label)
        else:
            correct = (true_label == pred_label).astype(float)
            accuracy = correct.sum() / correct.size 
            return accuracy.item()
        
    # precision score
    def precision_score_single(self, true_label, pred_label, class_num):
        true_label = torch.tensor(true_label).to(self.device)
        pred_label = torch.tensor(pred_label).to(self.device)
        
        precision = torch.zeros(class_num).to(true_label.device)
        for i in range(class_num):
            tp = ((pred_label == i) & (true_label == i)).sum().item()
            fp = ((pred_label == i) & (true_label != i)).sum().item()
            precision[i] = tp / (tp + fp) if tp + fp > 0 else 0
        return precision.mean().item()

    def precision_score_multi(self, true_label, pred_label, class_num):
        true_label = torch.tensor(true_label).to(self.device)
        pred_label = torch.tensor(pred_label).to(self.device)
        
        precision = torch.zeros(class_num).to(true_label.device)
        for i in range(class_num):
            tp = ((pred_label[:, i] == 1) & (true_label[:, i] == 1)).sum().item()
            fp = ((pred_label[:, i] == 1) & (true_label[:, i] == 0)).sum().item()
            precision[i] = tp / (tp + fp) if tp + fp > 0 else 0
        return precision.mean().item()

    def precision_score(self, true_label, pred_label, class_num, isMultiLabel):
        if isMultiLabel == False:
            return self.precision_score_single(true_label, pred_label, class_num)
        else:
            return self.precision_score_multi(true_label, pred_label, class_num)
      
    # recall score
    def recall_score_single(self, true_label, pred_label, class_num):
        true_label = torch.tensor(true_label).to(self.device)
        pred_label = torch.tensor(pred_label).to(self.device)
        
        recall = torch.zeros(class_num).to(true_label.device)
        for i in range(class_num):
            tp = ((pred_label == i) & (true_label == i)).sum().item()
            fn = ((pred_label != i) & (true_label == i)).sum().item()
            recall[i] = tp / (tp + fn) if tp + fn > 0 else 0
        return recall.mean().item()

    def recall_score_multi(self, true_label, pred_label, class_num):
        true_label = torch.tensor(true_label).to(self.device)
        pred_label = torch.tensor(pred_label).to(self.device)
        
        recall = torch.zeros(class_num).to(true_label.device)
        for i in range(class_num):
            tp = ((pred_label[:, i] == 1) & (true_label[:, i] == 1)).sum().item()
            fn = ((pred_label[:, i] == 0) & (true_label[:, i] == 1)).sum().item()
            recall[i] = tp / (tp + fn) if tp + fn > 0 else 0
        return recall.mean().item()

    def recall_score(self, true_label, pred_label, class_num, isMultiLabel):
        if isMultiLabel == False:
            return self.recall_score_single(true_label, pred_label, class_num)
        else:
            return self.recall_score_multi(true_label, pred_label, class_num)
        
    # f1 score
    def f1_score_single(self, true_label, pred_label, class_num):
        precision = self.precision_score_single(true_label, pred_label, class_num)
        recall = self.recall_score_single(true_label, pred_label, class_num)
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    def f1_score_multi(self, true_label, pred_label, class_num):
        precision = self.precision_score_multi(true_label, pred_label, class_num)
        recall = self.recall_score_multi(true_label, pred_label, class_num)
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    def f1_score(self, true_label, pred_label, class_num, isMultiLabel):
        if isMultiLabel == False:
            return self.f1_score_single(true_label, pred_label, class_num)
        else:
            return self.f1_score_multi(true_label, pred_label, class_num)
        
def train_bert(net, num_epochs, train_iter, topic_loss_func, sentiment_loss_func, optimizer, device):
    """train bert model"""

    epoch_loss = []
    epoch_topic_loss = []
    epoch_sentiment_loss = []

    epoch_topic_accuracy = []  
    epoch_topic_precision = []  
    epoch_topic_recall = []  
    epoch_topic_f1 = []  

    epoch_sentiment_accuracy = [] 
    epoch_sentiment_precision = []  
    epoch_sentiment_recall = []  
    epoch_sentiment_f1 = []  

    for epoch in range(num_epochs):
        net.train()  
        total_loss = 0.0
        total_topic_loss = 0.0
        total_sentiment_loss = 0.0
        correct_sentiment = 0
        total_sentiment = 0
        correct_topic = 0
        total_topic = 0
        
        # predicted labels of every epoch
        predicted_topics_epoch = []  
        predicted_sentiments_epoch = []  
        # true labels of every epoch
        y_epoch = []  
        z_epoch = [] 

        for X, y, z in train_iter:
            input_ids, attention_mask = X
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            y, z = y.to(device), z.to(device)

            topic_output, sentiment_output = net(input_ids, attention_mask)
            
            # calc loss
            topic_loss = topic_loss_func(topic_output, y.float())  
            sentiment_loss = sentiment_loss_func(sentiment_output, z)  
            loss = topic_loss + sentiment_loss
            
            # train model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_topic_loss += topic_loss.item()
            total_sentiment_loss += sentiment_loss.item()
            
            # predicted labels 
            _, predicted_sentiment = torch.max(sentiment_output, 1) 
            correct_sentiment += (predicted_sentiment == z).sum().item()
            total_sentiment += z.size(0)

            predicted_topic = (topic_output > 0.5).int() 
            correct_topic += (predicted_topic == y).sum().item() 
            total_topic += y.size(0) * y.size(1)  
                      
            predicted_topics_epoch.append(predicted_topic.cpu().numpy())      
            predicted_sentiments_epoch.append(predicted_sentiment.cpu().numpy())
            
            y_epoch.append(y.cpu().numpy())
            z_epoch.append(z.cpu().numpy())

        # loss of epoch
        avg_loss = total_loss / len(train_iter)
        avg_topic_loss = total_topic_loss / len(train_iter)
        avg_sentiment_loss = total_sentiment_loss / len(train_iter)

        # concatenate all result (by batch)
        predicted_topics_epoch = np.concatenate(predicted_topics_epoch, axis=0)    
        predicted_sentiments_epoch = np.concatenate(predicted_sentiments_epoch, axis=0)
        y_epoch = np.concatenate(y_epoch, axis=0)
        z_epoch = np.concatenate(z_epoch, axis=0)

        # calculate metrics
        topic_metrics = calc_metrics(y_epoch, predicted_topics_epoch, class_num=10, isMultiLabel=True, device = device)
        topic_metrics.compute_metrics()
        sentiment_metrics = calc_metrics(z_epoch, predicted_sentiments_epoch, class_num=3, isMultiLabel=False, device = device)  
        sentiment_metrics.compute_metrics()

        topic_accuracy = topic_metrics.accuracy_score
        topic_precision = topic_metrics.precision_score
        topic_recall = topic_metrics.recall_score
        topic_f1 = topic_metrics.f1_score

        sentiment_accuracy = sentiment_metrics.accuracy_score
        sentiment_precision = sentiment_metrics.precision_score
        sentiment_recall = sentiment_metrics.recall_score
        sentiment_f1 = sentiment_metrics.f1_score

        # metrics of every epoch
        epoch_loss.append(avg_loss)
        epoch_topic_loss.append(avg_topic_loss)
        epoch_sentiment_loss.append(avg_sentiment_loss)

        epoch_topic_accuracy.append(topic_accuracy)
        epoch_topic_precision.append(topic_precision)
        epoch_topic_recall.append(topic_recall)
        epoch_topic_f1.append(topic_f1)

        epoch_sentiment_accuracy.append(sentiment_accuracy)
        epoch_sentiment_precision.append(sentiment_precision)
        epoch_sentiment_recall.append(sentiment_recall)
        epoch_sentiment_f1.append(sentiment_f1)

        print(f"epoch {epoch+1}/{num_epochs}")
        print(f"  avg loss: {avg_loss:.4f}, topic loss: {avg_topic_loss:.4f}, sentiment loss: {avg_sentiment_loss:.4f}")
        print(f"  sentiment accuracy: {sentiment_accuracy:.4f}, topic accuracy: {topic_accuracy:.4f}")
        
    net_loss = (epoch_loss, epoch_topic_loss, epoch_sentiment_loss)
    topic_metrics = (epoch_topic_accuracy, epoch_topic_precision, epoch_topic_recall, epoch_topic_f1)
    sentiment_metrics = (epoch_sentiment_accuracy, epoch_sentiment_precision, epoch_sentiment_recall,epoch_sentiment_f1 )

    return net, net_loss, topic_metrics, sentiment_metrics


def visualize_epochs(epoch_loss, epoch_topic_loss, epoch_sentiment_loss,
                 epoch_topic_accuracy, epoch_topic_precision, epoch_topic_recall, epoch_topic_f1,
                 epoch_sentiment_accuracy, epoch_sentiment_precision, epoch_sentiment_recall, epoch_sentiment_f1):
    """visualization train epochs"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].plot(epoch_loss, label='Total Loss', color='blue', linewidth=2)
    axs[0].plot(epoch_topic_loss, label='Topic Loss', color='green', linewidth=2)
    axs[0].plot(epoch_sentiment_loss, label='Sentiment Loss', color='red', linewidth=2)
    axs[0].set_title('Loss Curves')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))  
    axs[0].grid(False)

    axs[1].plot(epoch_topic_accuracy, label='Topic Accuracy', color='blue', linewidth=2)
    axs[1].plot(epoch_topic_precision, label='Topic Precision', color='green', linewidth=2)
    axs[1].plot(epoch_topic_recall, label='Topic Recall', color='orange', linewidth=2)
    axs[1].plot(epoch_topic_f1, label='Topic F1 Score', color='red', linewidth=2)
    axs[1].set_title('Topic Metrics')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Metrics')
    axs[1].legend()
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))  # 只显示整数刻度
    axs[1].grid(False)

    axs[2].plot(epoch_sentiment_accuracy, label='Sentiment Accuracy', color='blue', linewidth=2)
    axs[2].plot(epoch_sentiment_precision, label='Sentiment Precision', color='green', linewidth=2)
    axs[2].plot(epoch_sentiment_recall, label='Sentiment Recall', color='orange', linewidth=2)
    axs[2].plot(epoch_sentiment_f1, label='Sentiment F1 Score', color='red', linewidth=2)
    axs[2].set_title('Sentiment Metrics')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Metrics')
    axs[2].legend()
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))  
    axs[2].grid(False)

    plt.tight_layout()
    os.makedirs('./result', exist_ok=True)
    plt.savefig('./result/training_metrics.png')
    plt.show()
    
def test_bert(net, test_iter, device):
    net.eval()

    correct_sentiment = 0
    total_sentiment = 0
    correct_topic = 0
    total_topic = 0

    test_predicted_topics_all = []  
    test_y_all = []  
    test_predicted_sentiments_all = []  
    test_z_all = []  

    # 不计算梯度
    with torch.no_grad():
        for X, y, z in test_iter:
            input_ids, attention_mask = X
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            y, z = y.to(device), z.to(device)

            topic_output, sentiment_output = net(input_ids, attention_mask)

            _, predicted_sentiment = torch.max(sentiment_output, 1) 
            correct_sentiment += (predicted_sentiment == z).sum().item()
            total_sentiment += z.size(0)

            predicted_topic = (topic_output > 0).int() 
            correct_topic += (predicted_topic == y).sum().item() 
            total_topic += y.size(0)   

            test_predicted_topics_all.append(predicted_topic.cpu().numpy())
            test_y_all.append(y.cpu().numpy())
            test_predicted_sentiments_all.append(predicted_sentiment.cpu().numpy())
            test_z_all.append(z.cpu().numpy())

    test_predicted_topics_all = np.concatenate(test_predicted_topics_all, axis=0)
    test_y_all = np.concatenate(test_y_all, axis=0)
    test_predicted_sentiments_all = np.concatenate(test_predicted_sentiments_all, axis=0)
    test_z_all = np.concatenate(test_z_all, axis=0)

    test_topic_metrics = calc_metrics(test_y_all, test_predicted_topics_all, class_num=10, isMultiLabel=True, device=device)
    test_topic_metrics.compute_metrics()
    test_sentiment_metrics = calc_metrics(test_z_all, test_predicted_sentiments_all, class_num=3, isMultiLabel=False, device=device)  
    test_sentiment_metrics.compute_metrics()

    test_topic_accuracy = test_topic_metrics.accuracy_score
    test_topic_precision = test_topic_metrics.precision_score
    test_topic_recall = test_topic_metrics.recall_score
    test_topic_f1 = test_topic_metrics.f1_score

    test_sentiment_accuracy = test_sentiment_metrics.accuracy_score
    test_sentiment_precision = test_sentiment_metrics.precision_score
    test_sentiment_recall = test_sentiment_metrics.recall_score
    test_sentiment_f1 = test_sentiment_metrics.f1_score

    print(f"test topic accuracy: {test_topic_accuracy:.4f}")
    print(f"test topic precision: {test_topic_precision:.4f}")
    print(f"test topic recall: {test_topic_recall:.4f}")
    print(f"test topic f1 score: {test_topic_f1:.4f}")

    print(f"test sentiment accuracy: {test_sentiment_accuracy:.4f}")
    print(f"test sentiment precision: {test_sentiment_precision:.4f}")
    print(f"test sentiment recall: {test_sentiment_recall:.4f}")
    print(f"test sentiment f1 Score: {test_sentiment_f1:.4f}")