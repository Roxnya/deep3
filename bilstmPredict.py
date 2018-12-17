import random
import sys
import dynet as dy
import pickle
import numpy as np
import time

epochs = 5
iteration_till_dev_size = 500 #num sentences
hid_layer = 50
top_hidden_layer = 60
out_layer = 5
tags = ()
EMB_SIZE = 50
BILSTM_INPUT = 50
CHAR_EMB_SIZE = 30
unk_word = "*UNK*"

class LstmAcceptor(object):
    def __init__(self, in_dim, lstm_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        outputs = lstm.transduce(sequence)
        return  outputs[-1]

class BiLstm(object):
    def __init__(self, repr, emb_dim, hidden_dim, top_hidd_dim, model, out_dim):
        self.repr = repr
        self.model = model
        self.forward_top_builder = dy.VanillaLSTMBuilder(1, 2*hidden_dim, top_hidd_dim, model)
        self.forward_bot_builder = dy.VanillaLSTMBuilder(1, emb_dim, hidden_dim, model)
        self.backward_top_builder = dy.VanillaLSTMBuilder(1, 2*hidden_dim, top_hidd_dim, model)
        self.backward_bot_builder = dy.VanillaLSTMBuilder(1, emb_dim, hidden_dim, model)
        self.W = self.model.add_parameters((out_dim, 2*top_hidd_dim), init="uniform", scale = 0.8)
        if repr == "d":
            self.U = self.model.add_parameters((BILSTM_INPUT, BILSTM_INPUT+CHAR_EMB_SIZE), init="uniform", scale = 0.8)
            self.lstm = LstmAcceptor(CHAR_EMB_SIZE, BILSTM_INPUT, self.model)
        if repr == "b":
            self.lstm = LstmAcceptor(CHAR_EMB_SIZE, BILSTM_INPUT, self.model)
    def __call__(self, sequence):
        forward_top_lstm = self.forward_top_builder.initial_state()
        forward_bot_lstm = self.forward_bot_builder.initial_state()
        backward_top_lstm = self.backward_top_builder.initial_state()
        backward_bot_lstm = self.backward_bot_builder.initial_state()
        #W = self.W.expr() # convert the parameter into an Expession (add it to graph)
        W = dy.parameter(self.W)
        sequence = self.__get_vec_by_rep__(sequence)
        f_outputs = forward_bot_lstm.transduce(sequence)
        b_outputs = backward_bot_lstm.transduce(sequence[::-1])
        new_outs = []
        for f, b in zip(f_outputs, reversed(b_outputs)):
            new_outs.append(dy.concatenate([f, b]))
        f_outputs = forward_top_lstm.transduce(new_outs)
        b_outputs = backward_top_lstm.transduce(new_outs[::-1])
        new_outs = []
        for f, b in zip(f_outputs, reversed(b_outputs)):
            new_outs.append(dy.softmax((W * dy.concatenate([f, b]))))
        return new_outs

    def __get_vec_by_rep__(self, sequence):
        seq_rep = []
        for w in sequence:
            if self.repr == "a":
                seq_rep.append(embeds[voc[w]])
            elif self.repr == "b":
                vecs = [embeds[voc[char]] for char in w]
                res = self.lstm(vecs)
                seq_rep.append(res)
            elif self.repr == "c":
                w_embed = embeds[voc[w[1]]]
                sub_word = w_embed+w_embed+w_embed if len(w[1]) <= 3 else embeds[voc[w[0]]]+w_embed+embeds[voc[w[2]]]
                seq_rep.append(sub_word)
            elif self.repr == "d":
                U = self.U.expr()
                vecs = [embeds[voc[char]] for char in w[1]]
                char_out = self.lstm(vecs)
                seq_rep.append(U*(dy.concatenate([embeds[voc[w[0]]], char_out])))
        return seq_rep

def test():
    words_so_far = 0
    preds = []
    for sequence, labels in test_set:
        dy.renew_cg()
        preds = bilstm(sequence)

        for (pred,label) in zip(preds,labels):
            preds.append(np.argmax(pred.npvalue()))
    return preds

def init_params_by_dataset(data_set):
    global out_layer
    out_layer = len(tags)

def init_params_by_rep(rep, voc):
    if rep == "a":
        _, test_set = build_a_rep(data_set+"/test", True, voc)
    elif rep == "b":
        _, test_set = build_b_rep(data_set + "/test", True, voc)
    elif rep == "c":
        _, test_set = build_c_rep(data_set + "/test", True, voc)
    elif rep == "d":
        _, test_set = build_d_rep(data_set + "/test", True, voc)
    voc[unk_word] = len(voc)
    embeds = m.add_lookup_parameters((len(voc), EMB_SIZE), init="normal", mean = 0, std = 1)
    #embeds = m.add_lookup_parameters((len(voc), EMB_SIZE))
    return voc, embeds, test_set

#-----------------Representation REGION
def build_a_rep(trainFile, test, train_voc=None):
    voc, examples = build_vocab(trainFile, vocab_by_word, test, train_voc)
    return voc, examples

def build_b_rep(trainFile, test, train_voc=None):
    global CHAR_EMB_SIZE, EMB_SIZE
    EMB_SIZE = CHAR_EMB_SIZE
    voc, examples = build_vocab(trainFile, vocab_by_letter, test, train_voc)
    return voc, examples

def build_c_rep(trainFile, test, train_voc=None):
    voc, examples = build_vocab(trainFile, vocab_by_sub_word, test, train_voc)
    return voc, examples

def build_d_rep(trainFile, test, train_voc=None):
    global CHAR_EMB_SIZE, EMB_SIZE
    EMB_SIZE = CHAR_EMB_SIZE
    voc, examples = build_vocab(trainFile, vocab_by_word_letter, test, train_voc)
    return voc, examples

#-----------------vocab add functions REGION
def vocab_by_letter(voc, word, tag, test = False):
    examples = []
    for w in word:
        if w not in voc:
            if test:
                w = unk_word
            else:
                voc[w] = len(voc)
        examples.append(w)
    return [examples]

def vocab_by_sub_word(voc,word,tag, test = False):
    if word not in voc:
        if test:
            word = unk_word
        else:
            voc[word] = len(voc)
    if len(word) <= 3:
        return [(word,word,word)]
    pre = word[0:3]
    if word[0:3] not in voc:
        if test:
            pre = unk_word
        else:
            voc[pre] = len(voc)

    post = word[len(word) - 3:len(word)]
    if word[len(word) - 3:len(word)] not in voc:
        if test:
            post = unk_word
        else:
            voc[post] = len(voc)
    return [(pre, word, post)]


def vocab_by_word_letter(voc, word, tag, test = False):
    w_rep = vocab_by_word(voc, word, tag, test)[0]
    char_rep = vocab_by_letter(voc, word, tag, test)[0]
    return [(w_rep, char_rep)]

def vocab_by_word(voc, word, tag, test = False):
    if word not in voc:
        if test:
            word = unk_word
        else:
            voc[word] = len(voc)
    return [word]

def build_vocab(trainFile, vocab_by_word, test = False, train_voc = None):
    voc = dict() if not test else train_voc
    examples = []
    sentence = []
    sent_tags = []
    with open(trainFile, "r") as f:
        content = f.readlines()
    for line in content:
        if not line.isspace():
            word = line.split()
            ex = vocab_by_word(voc, word, def_tag, test)
            sentence.extend(ex)
            sent_tags.append(tags[def_tag])
        else:
            examples.append((sentence, sent_tags))
            sentence=[]
            sent_tags=[]
    return voc, examples

def write_test(pred):
    out = open(inputFile + "." + data_set, "w")
    delim = " " if data_set == "pos" else "\t"
    with open(data_set + "/test", "r") as f:
        content = f.readlines()
    i = 0
    for line in content:
        if not line.isspace():
            out.write(line[:-1] + delim + pred_to_word[int(pred[i])] + "\n")
            i += 1
    out.close()

if __name__ == '__main__':
    global tags
    if len(sys.argv) != 7:
        print("Program expects exactly 4 arguments, representation, train file, model file and data set type")
        exit(-1)

    add,  repr, modelFile, inputFile, data_set, tags_file, voc_file = sys.argv
    with open(voc_file,"rb") as f:
        voc = pickle.load(f)
    with open(tags_file, "rb") as f:
        tags = pickle.load(f)

    pred_to_word = {v: k for k, v in tags.items()}
    def_tag = pred_to_word[0]
    m = dy.Model()
    trainer = dy.AdamTrainer(m)
    init_params_by_dataset(data_set)
    voc, embeds, test_set = init_params_by_rep(repr, voc)
    bilstm = BiLstm(repr, BILSTM_INPUT, hid_layer, top_hidden_layer, m, out_layer)
    m.populate(modelFile+".model")
    bilstm.forward_top_builder.param_collection().populate(modelFile + "_bi_f_top.model")
    bilstm.forward_bot_builder.param_collection().populate(modelFile + "_bi_f_bot.model")
    bilstm.backward_top_builder.param_collection().populate(modelFile + "_bi_b_top.model")
    bilstm.backward_bot_builder.param_collection().populate(modelFile + "_bi_b_bot.model")
    bilstm.lstm.builder.param_collection().populate(modelFile + "_lstm.model")
    preds = test()
    write_test(preds)