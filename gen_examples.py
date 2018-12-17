import random
import pickle

ex_size = 9

def generate_examples(num_ex, alphabet, label):
    ex,examples = [], []
    for _ in range(num_ex):
        j = 0
        for ex_num in range(0,ex_size-1,2):
            ex.append("".join([random.choice("123456789") for i in range(random.randint(1, 21))]))
            ex.append("".join([random.choice(alphabet[j]) for i in range(random.randint(1, 21))]))
            j+=1
        ex.append("".join([random.choice("123456789") for i in range(random.randint(1, 21))]))
        examples.append(("".join(ex), label))
        ex = []
    return examples

def generate_train_test_set():
    pos_examples = generate_examples(1200, ['a','b','c','d'],0)
    neg_examples = generate_examples(1200, ['a','c','b','d'],1)
    with open("train_set","wb") as f:
        pickle.dump(pos_examples[0:600]+neg_examples[0:600],f)
    with open("test_set","wb") as f:
        pickle.dump(pos_examples[600:1200]+neg_examples[600:1200],f)

def generate_anbn_examples():
    examples = []
    for _ in range(1000):
        #choose if center repeats or not
        a ="".join(['a' for i in range(random.randint(1, 60))])
        b = "".join(['b' for i in range(len(a))])
        ab = random.shuffle(a+b)
        examples.append((ab, 0))
        ra = "".join(['a' for i in range(random.randint(1, 60))])
        b_len = random.randint(1, 60)
        while b_len == len(ra):
            b_len = random.randint(1, 60)
        rb = "".join(['b' for i in range(b_len)])
        rab = random.shuffle(rb+ra)
        examples.append((rab, 1))
    return examples

def save_anbn():
    train = generate_anbn_examples()
    test = generate_anbn_examples()
    random.shuffle(train)
    random.shuffle(test)
    with open("anbn/train_set", "wb+") as f:
        pickle.dump(train, f)
    with open("anbn/test_set", "wb+") as f:
        pickle.dump(test, f)

#generate_train_test_set()
