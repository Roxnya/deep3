import random
import pickle

ex_size = 9

def generate_examples(num_ex, alphabet, label):
    ex,examples = [], []
    for _ in range(num_ex):
        # num_sequence = [np.random.randint(1, 10, np.random.randint(1, 21)) for i in range(5)]
        # num_letters = [[np.random.choise(alphabet(i), np.random.randint(1, 21)) for i in range(4)]]
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
generate_train_test_set()