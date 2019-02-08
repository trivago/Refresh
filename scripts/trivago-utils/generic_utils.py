from nltk.tokenize import word_tokenize



def build_vocab_dictionary(vocab_file):
    print("*** Building vocab dictionary")
    # Build vocab dictionary
    vocab_dict = {}
    for line_count, line in enumerate(vocab_file):
        vocab_dict[str(line.rstrip("\n"))] = line_count

    print("*** Vocabulary size: {}".format(len(vocab_dict)))
    return vocab_dict

def convert_sentence_to_word_ids(sentence, vocab_dict):
    word_id_sent = []

    word_tokenized_sent = word_tokenize(sentence)

    for word_token in word_tokenized_sent:
        if word_token in vocab_dict:
            word_id_value = vocab_dict[word_token]
            word_id_sent.append(str(word_id_value))
        else:
            word_id_sent.append(str(1))

    return word_id_sent

