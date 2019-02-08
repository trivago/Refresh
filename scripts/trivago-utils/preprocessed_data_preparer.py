# preprocessed_data_preparer.py -- Prepares trivago USP corpora for the Refresh model when called.
# Requires Python 3 and Tensorflow v1+ to run.
# @author Saad Mahamood

import argparse
import os
import os.path
from nltk.tokenize import sent_tokenize
import uuid

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy import spatial

from generic_utils import build_vocab_dictionary
from generic_utils import convert_sentence_to_word_ids


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input file to process', required=True)
    parser.add_argument('-t', '--target', help="Corresponding target file", required=True)
    parser.add_argument('-v', '--vocab', help="Vocab file", required=True)
    parser.add_argument('-dt', '--data_type', help="Data Type e.g. training, test, etc.", required=True)

    args = parser.parse_args()

    input_argument = args.input
    target_argument = args.target
    vocab_argument = args.vocab
    data_type_argument = args.data_type

    if os.path.isfile(input_argument) and os.path.isfile(target_argument) \
            and os.path.isfile(vocab_argument):
        print("*** Opening file...")

        input_file_obj = open(input_argument, "r")
        target_file_obj = open(target_argument, "r")
        vocab_file_obj = open(vocab_argument, "r")

        target_sentence_ids = create_titles_file(target_file_obj, vocab_file_obj, data_type_argument)

        create_image_file(target_file_obj, vocab_file_obj, target_sentence_ids, data_type_argument)

        create_single_oracle(input_file_obj, target_sentence_ids, data_type_argument)
    else:
        print("*** Invalid arguments provided.")

def create_titles_file(target_file, vocab_file, data_type_argument):
    target_sentences_dict = {}
    print("*** Creating Title file...")
    title_file = open(os.path.dirname(os.path.realpath(target_file.name)) +
                      "/usp-" + data_type_argument + ".title", "w")

    vocab_dict = build_vocab_dictionary(vocab_file)

    for line_count, line in enumerate(target_file):
        title_uuid = "usp-" + uuid.uuid4().hex
        target_sentences_dict[title_uuid] = line
        word_id_sent = convert_sentence_to_word_ids(line, vocab_dict)
        title_file.write(title_uuid + "\n")
        title_file.write(" ".join(word_id_sent) + "\n\n")

    title_file.close()
    print("*** Finished Title file...")

    return target_sentences_dict


def create_image_file(input_file, vocab_file, target_sentence_dict, data_type_argument):
    print("*** Creating Image file...")
    # Build vocab dictionary
    vocab_dict = build_vocab_dictionary(vocab_file)

    image_file = open(os.path.dirname(os.path.realpath(input_file.name)) +
                      "/usp-" + data_type_argument + ".image", "w")

    target_sentence_ids = list(target_sentence_dict.keys())
    for title_id in target_sentence_ids:
        image_file.write(title_id + "\n")
        for line_count, line in enumerate(input_file):
            word_id_sent = convert_sentence_to_word_ids(line, vocab_dict)
            image_file.write(" ".join(word_id_sent) + "\n\n")
    image_file.close()
    print("*** Finished Image file...")


def create_single_oracle(input_file, target_sentence_dict, data_type_argument):
    print("*** Creating Single Oracle file...")
    oracle_file =  open(os.path.dirname(os.path.realpath(input_file.name)) +
                      "/usp-" + data_type_argument + ".label.singleoracle", "w")

    target_sentence_ids = list(target_sentence_dict.keys())
    for line_count, line in enumerate(input_file):
        if line_count <= (len(target_sentence_ids) - 1):
            title_uuid = target_sentence_ids[line_count]
            target_sentence = target_sentence_dict[title_uuid]

            oracle_file.write(title_uuid + "\n")

            print("*** Processing line number {}".format(line_count))
            sent_tokenize_list = sent_tokenize(line)
            print("*** Number of sentences: {}".format(len(sent_tokenize_list)))
            print("*** Current Progress: {}".format(str((line_count / len(input_file)) * 100)) + "%")

            distances = prepare_labels(sent_tokenize_list, target_sentence)

            for dist_count, a_distance in enumerate(distances):
                oracle_file.write(str(int(a_distance)) + "\n")
                if dist_count == (len(distances) - 1):
                    oracle_file.write("\n")
    oracle_file.close()





def prepare_labels(source_sentences:list, target:str, topk:int=3):
    # function returning ids of topk number of best sentences
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        input_sentences = tf.placeholder(tf.string, shape=[None])
        embeddings = embed(input_sentences)

        embeds = embeddings.eval(feed_dict={input_sentences: [target] + source_sentences })
        distances = []

        target = None
        matrix = np.array(embeds)
        for i, w in enumerate(matrix):
            if i == 0:
                # remember the target vector
                target = w
            else:
                # store simmilarity measures betwean each of the sentences and the target
                distances.append(spatial.distance.cosine(target, w))
        distances = np.array(distances)
        distances = distances.argsort()[-topk:][::-1]

        target_binarized = np.zeros(len(source_sentences))
        np.put(target_binarized, distances, 1, mode='clip')

        return target_binarized.tolist()


def main():
    parse_arguments()


if __name__ == '__main__':
    main()
