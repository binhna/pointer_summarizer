from torchtext.data import Field, Example, Dataset



def read_vocabs(vocab_file):
    vocabs = []
    with open(vocab_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        line_split = line.split()
        if len(line_split) == 2:
            vocabs.append(line_split[0])
    return vocabs

def Mydataset(Dataset):
    def __init__(self, data, fields):
        super(Mydataset, self).__init__(
            [Example.fromlist([d['src'], d['tgt']], fields) for d in data],
            fields
        )


if __name__ == "__main__":
    pass
    # vocabs = read_vocabs('../data/finished_files/vocab')
    # print(len(vocabs))