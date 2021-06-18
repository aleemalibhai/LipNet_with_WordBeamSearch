from num2words import num2words

with open('/home/aalibhai/Thesis/LipNet-CTCWBS/LRS3_corpus.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        words = line.split(' ')
        for word in words:
            if word.isnumeric():
                word = num2words(word)
            print (word)