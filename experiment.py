import dynet as dy
import numpy as np
import pickle
from sklearn.utils import shuffle

layers = 2
VOC_SIZE = 13
EMB_SIZE = 100
out_layer = 2
hid_layer = 100
class LstmAcceptor(object):
    def __init__(self, in_dim, lstm_dim, out_dim, hid_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W = model.add_parameters((hid_dim, lstm_dim))
        self.V = model.add_parameters((out_dim, hid_dim))
    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W = self.W.expr() # convert the parameter into an Expession (add it to graph)
        V = self.V.expr()
        outputs = lstm.transduce(sequence)
        result = V*dy.tanh(W * outputs[-1])
        return result

def train(set, epochs, val = False):
    sum_of_losses = 0.0
    correct = 0.0
    print("Performing train")
    for epoch in range(epochs):
        for sequence, label in set:
            dy.renew_cg()  # new computation graph
            vecs = [embeds[voc_map[char]] for char in sequence]
            preds = acceptor(vecs)
            y_hat = np.argmax(preds.npvalue())
            correct += 1 if y_hat == label else 0
            loss = dy.pickneglogsoftmax(preds, label)
            sum_of_losses += loss.npvalue()
            if not val:
                loss.backward()
                trainer.update()
        print (sum_of_losses / len(set))
        print (correct/len(set) * 100)
        sum_of_losses = 0.0
        correct = 0.0

def test(set):
    # prediction code:
    for sequence in set:
        dy.renew_cg()  # new computation graph
        vecs = [embeds[i] for i in sequence]
        preds = dy.softmax(acceptor(vecs))
        vals = preds.npvalue()
        print (np.argmax(vals), vals)

def get_data():
    with open("voc_map", "rb") as f:
        voc_map = pickle.load(f)
    with open("train_set", "rb") as f:
        train = pickle.load(f)
    train = shuffle(train)
    train_size = round(0.8*len(train))
    train_set = train[:train_size]
    val_set = train[train_size:]

    with open("test_set2", "rb") as f:
        test_set = pickle.load(f)
    return voc_map, train_set, val_set, test_set

m = dy.Model()
voc_map, train_set, val_set, test_set = get_data()
epochs = 20
#use default rate
trainer = dy.AdamTrainer(m)
embeds = m.add_lookup_parameters((VOC_SIZE, EMB_SIZE))
acceptor = LstmAcceptor(EMB_SIZE, 100, out_layer, hid_layer, m)
train(train_set, epochs)
train(val_set, 1, True)
train(test_set, 1, True)