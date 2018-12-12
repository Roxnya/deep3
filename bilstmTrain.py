import sys
import dynet as dy
import pickle
import numpy as np

epochs = 5
batch_size = 500 #num sentences
hid_layer = 300
out_layer = 5
tags = ()
EMB_SIZE = 50

class BiLstm(object):
    def __init__(self, repr, emb_dim, hidden_dim, model, out_dim):
        self.repr = repr
        self.model = model
        self.forward_top_builder = dy.VanillaLSTMBuilder(1, 2*hidden_dim, hidden_dim, model)
        self.forward_bot_builder = dy.VanillaLSTMBuilder(1, emb_dim, hidden_dim, model)
        self.backward_top_builder = dy.VanillaLSTMBuilder(1, 2*hidden_dim, hidden_dim, model)
        self.backward_bot_builder = dy.VanillaLSTMBuilder(1, emb_dim, hidden_dim, model)
        self.W = self.model.add_parameters((out_dim, 2*hidden_dim))
        if repr == "d":
            self.U = self.model.add_parameters((emb_dim, emb_dim))
    def __call__(self, sequence):
        forward_top_lstm = self.forward_top_builder.initial_state()
        forward_bot_lstm = self.forward_bot_builder.initial_state()
        backward_top_lstm = self.backward_top_builder.initial_state()
        backward_bot_lstm = self.backward_bot_builder.initial_state()
        W = self.W.expr() # convert the parameter into an Expession (add it to graph)
        U = self.U.expr() if repr == "d" else None
        sequence = self.__get_vec_by_rep__(sequence, U)
        f_outputs = forward_bot_lstm.transduce(sequence)
        b_outputs = backward_bot_lstm.transduce(sequence[::-1])
        new_outs = []
        for f, b in zip(f_outputs, reversed(b_outputs)):
            new_outs.append(dy.concatenate([f, b]))
        f_outputs = forward_top_lstm.transduce(new_outs)
        b_outputs = backward_top_lstm.transduce(new_outs[::-1])
        new_outs = []
        for f, b in zip(f_outputs, reversed(b_outputs)):
            new_outs.append(dy.concatenate([f, b]))
        outs = []
        for i in range(len(sequence)):
            outs.append(W*(new_outs[i]))
        return outs

    def __get_vec_by_rep__(self, sequence, U = None):
        seq_rep = []
        for w in sequence:
            if self.repr == "a":
                seq_rep.append(embeds[voc[w]])
            elif self.repr == "b":
                seq_rep.extend([embeds[voc[char]] for char in w])
            elif self.repr == "c":
                w_embed = embeds[voc[w]]
                sub_word = (w_embed, w_embed, w_embed) if len(w) <= 3 else (
                embeds[voc[w[0:3]]], w_embed, embeds[voc[w[len(w) - 3:len(w)]]])
                seq_rep.append(sub_word)
            elif self.repr == "d":
                seq_rep.append(U*(embeds[voc[w]]+[embeds[voc[char]] for char in w]))
        return seq_rep

def train(set, epochs, val = False):
    sum_of_losses = 0.0
    correct = 0.0
    sentence_idx = 0
    print("Performing train")
    for epoch in range(epochs):
        # for j in range(int(len(set) / 500)):
        #     mini_set = set[j*batch_size:j*batch_size + batch_size]
        for sequence, labels in set:
            dy.renew_cg()
            #dy.renew_cg()  # new computation graph
            preds = bilstm(sequence)
            local_losses = []
            for i,pred in enumerate(preds):
                y_hat = np.argmax(dy.softmax(pred).npvalue())
                correct += 1 if y_hat == labels[i] else 0
                loss = dy.pickneglogsoftmax(pred, labels[i])
                local_losses.append(loss)
            if not val:
                sent_loss = dy.esum(local_losses)#/len(sequence)
                sum_of_losses += sent_loss.scalar_value()
                sent_loss.backward()
                trainer.update()
            if sentence_idx == batch_size:
                print(sum_of_losses / len(set))
                print(correct / len(set) * 100)
                sentence_idx=0
                #sum_of_losses = 0.0
                #correct = 0.0
                #dy.renew_cg()
            sentence_idx +=1
        print (sum_of_losses / len(set))
        print (correct/len(set) * 100)
        sum_of_losses = 0.0
        correct = 0.0


def init_params_by_dataset(data_set):
    global out_layer, tags
    tags_file = "ner_tags" if data_set == "ner" else "pos_tags"
    with open(tags_file, "rb") as f:
        tags = pickle.load(f)
    out_layer = len(tags)

def init_params_by_rep(rep, trainFile):
    if rep == "a":
        voc, examples = build_a_rep(trainFile)
    elif rep == "b":
        voc, examples = build_b_rep(trainFile)
    elif rep == "c":
        voc, examples = build_c_rep(trainFile)
    elif rep == "d":
        voc, examples = build_d_rep(trainFile)
    voc["*UNK*"] = len(voc)
    embeds = m.add_lookup_parameters((len(voc), EMB_SIZE))
    return voc, embeds, examples

#-----------------Representation REGION
def build_a_rep(trainFile):
    voc, examples = build_vocab(trainFile, vocab_by_word)
    return voc, examples

def build_b_rep(trainFile):
    global EMB_SIZE
    EMB_SIZE = 20
    voc, examples = build_vocab(trainFile, vocab_by_letter)
    return voc, examples

def build_c_rep(trainFile):
    voc, examples = build_vocab(trainFile, vocab_by_sub_word)
    return voc, examples

def build_d_rep(trainFile):
    voc, examples = build_vocab(trainFile, vocab_by_word_letter)
    return voc, examples

#-----------------vocab add functions REGION
def vocab_by_letter(voc, word, tag, test = False):
    examples = []
    for w in word:
        if word not in voc and not test:
            voc[word] = len(voc)
        if test:
            w = "*UNK*"
        examples.append(w)
    return examples

def vocab_by_sub_word(voc,word,tag, test = False):
    if word not in voc and not test:
        voc[word] = word
    if test:
        word = "*UNK*"
    if len(word) <= 3:
        return ((word,word,word),tag)
    pre = word[0:3]
    if word[0:3] not in voc:
        if test:
            pre = "*UNK*"
        else:
            voc[word[0:3]] = len(voc)

    post = word[len(word) - 3:len(word)]
    if word[len(word) - 3:len(word)] not in voc:
        if test:
            post = "*UNK*"
        else:
            voc[word[len(word) - 3:len(word)]] = len(voc)
    return [(pre, word, post)]


def vocab_by_word_letter(voc, word, tag, test = False):
    w_rep = vocab_by_word(voc, word, tag, test = False)[0]
    char_rep = (w for w,t in vocab_by_letter(voc, word, tag, test = False))
    return [(w_rep + char_rep)]

def vocab_by_word(voc, word, tag, test = False):
    if word not in voc and not test:
        voc[word] = len(voc)
    if test:
        word = "*UNK*"
    return [word]

def build_vocab(trainFile, vocab_by_word, test = False):
    voc = dict()
    examples = []
    sentence = []
    sent_tags = []
    with open(trainFile, "r") as f:
        content = f.readlines()
    for line in content:
        if not line.isspace():
            word, tag = line.split()
            ex = vocab_by_word(voc, word, tag, test)
            sentence.extend(ex)
            sent_tags.extend(np.repeat(tags[tag], len(ex)))
        else:
            examples.append((sentence, sent_tags))
            sentence=[]
    return voc, examples

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Program expects exactly 4 arguments, representation, train file, model file and data set type")
        exit(-1)

    add, repr, trainFile, modelFile, data_set = sys.argv
    m = dy.Model()
    trainer = dy.AdamTrainer(m, 0.01)
    init_params_by_dataset(data_set)
    voc, embeds, examples = init_params_by_rep(repr, trainFile)
    bilstm = BiLstm(repr, EMB_SIZE, hid_layer, m, out_layer)
    train(examples,epochs)