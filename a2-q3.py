#!/usr/bin/env python
# coding: utf-8

# In[1]:




def main():
    import nltk
    from nltk.tag import hmm
    from nltk.probability import LaplaceProbDist, MLEProbDist, ConditionalFreqDist, ConditionalProbDist, \
        LidstoneProbDist
    import sys
    import re

    characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ',',','.']
    char_dict = {
        'a':0,
        'b':1,
        'c':2,
        'd':3,
        'e':4,
        'f':5,
        'g':6,
        'h':7,
        'i':8,
        'j':9,
        'k':10,
        'l':11,
        'm':12,
        'n':13,
        'o':14,
        'p':15,
        'q':16,
        'r':17,
        's':18,
        't':19,
        'u':20,
        'v':21,
        'w':22,
        'x':23,
        'y':24,
        'z':25,
        ' ':26,
        ',':27,
        '.':28
    }

    def preprocess(text):
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"!", "!", text)
        text = re.sub(r"\/", "", text)
        text = re.sub(r"\^", "^", text)
        text = re.sub(r"\+", "+", text)
        text = re.sub(r"\-", "-", text)
        text = re.sub(r"\=", "=", text)
        text = re.sub(r'[0-9\s]',"=",text)
        text = re.sub(r"'", "=", text)
        text = re.sub(r"^https?:\/\/.*[\r\n]*","", text)
        text = re.sub(r'[^.,a-zA-Z0-9\s]', ' ',text)
        text = re.sub(r'  ',"",text)
        return text




    LAPLACE = False
    LM = False

    if "-laplace" in sys.argv:
        LAPLACE = True
    if "-lm" in sys.argv:
        LM = True

    directory = sys.argv[-1]

    print(directory + "\\train_plain.txt")
    file1 = open(directory + "\\train_plain.txt","r")
    file1_train_plain = file1.readlines()
    file1_train_plain = [line.lower().strip('\n') for line in file1_train_plain]
    file1_train_plain = [list(line) for line in file1_train_plain]
    file1.close()

    file1 = open(directory + "/train_cipher.txt","r")
    file1_train_cipher = file1.readlines()
    file1_train_cipher = [line.lower().strip('\n') for line in file1_train_cipher]
    file1_train_cipher = [list(line) for line in file1_train_cipher]
    file1.close()

    for_tagging = []
    for i in range(len(file1_train_cipher)):    #(cipher,plaintext)
        partial_list = []
        for j in range(len(file1_train_cipher[i])):
                partial_list.append((file1_train_cipher[i][j],file1_train_plain[i][j]))
        for_tagging.append(partial_list)

    #for_tagging = [tup for tup in for_tagging]
    #print(for_tagging)


    # In[114]:


    def estimator(fd, bins):
        return LaplaceProbDist(fd,bins)

    # load the new corpus
    new_file = open("TheStory.txt", "r", encoding="utf8")
    new_file = new_file.readlines()
    new_file = [line.lower().strip('\n').strip() for line in new_file]
    new_file = [line for line in new_file if len(line) > 1]
    new_corpus = [preprocess(line.strip()) for line in new_file if len(line) > 1]

    final_corpus1 = ""
    for line in new_corpus:
        final_corpus1 += line

    final_corpus2 = final_corpus1[1:]
    final_pairs = []
    for i in range(len(final_corpus1) - 1):
        final_pairs.append((final_corpus1[i], final_corpus2[i]))

    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = 0
    if LAPLACE:
        tagger = trainer.train_supervised(for_tagging,estimator)
        if LM:
            cfdist = ConditionalFreqDist(tup for tup in final_pairs)
            cpd = ConditionalProbDist(cfdist,estimator, 29)
            attribute = tagger.__getattribute__("_transitions")
            for (cond, prob_dist) in cpd.items():
                new_fd = prob_dist.freqdist() + attribute[cond].freqdist()
                if type(prob_dist) is LaplaceProbDist:
                    cpd[cond] = LaplaceProbDist(new_fd, prob_dist._bins)
                elif type(prob_dist) is MLEProbDist:
                    cpd[cond] = MLEProbDist(new_fd)
            tagger.__getattribute__('_transitions').update(cpd)
    elif LM:
        tagger = trainer.train_supervised(for_tagging)
        cfdist = ConditionalFreqDist(tup for tup in final_pairs)
        cpd = ConditionalProbDist(cfdist,estimator, 29)
        attribute = tagger.__getattribute__("_transitions")
        for (cond, prob_dist) in cpd.items():
            new_fd = prob_dist.freqdist() + attribute[cond].freqdist()
            if type(prob_dist) is LaplaceProbDist:
                cpd[cond] = LaplaceProbDist(new_fd, prob_dist._bins)
            elif type(prob_dist) is MLEProbDist:
                cpd[cond] = MLEProbDist(new_fd)
        tagger.__getattribute__('_transitions').update(cpd)
    else:
        tagger = trainer.train_supervised(for_tagging)


    # In[115]:


    file1 = open(directory + "/test_plain.txt","r")
    file1_test_plain = file1.readlines()
    file1_test_plain = [line.lower().strip('\n') for line in file1_test_plain]
    file1_test_plain = [list(line) for line in file1_test_plain]
    file1.close()

    file1 = open(directory + "/test_cipher.txt","r")
    file1_test_cipher = file1.readlines()
    file1_test_cipher = [line.lower().strip('\n') for line in file1_test_cipher]
    file1.close()

    tagged_result = []

    for i in range(len(file1_test_cipher)):
        tagged_list = tagger.tag(file1_test_cipher[i])
        tagged_result.append(tagged_list)

    file1_test_plain = [list(line) for line in file1_test_plain]

    char_counter = 0
    correct_counter = 0
    decipherment = ""
    for i in range(len(tagged_result)):
        for j in range(len(tagged_result[i])):
            char_counter += 1
            decipherment += tagged_result[i][j][1]
            if tagged_result[i][j][1] == file1_test_plain[i][j]:
                correct_counter += 1

    # print(correct_counter)
    # print(char_counter)
    print("The deciphered text:", decipherment)
    print("The decipherment accuracy:", correct_counter / char_counter)

if __name__=="__main__":
    main()
