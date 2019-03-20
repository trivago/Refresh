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

        vocab_dict = build_vocab_dictionary(open(vocab_argument, "r"))

        target_sentence_ids = create_titles_file(open(target_argument, "r"), vocab_dict, data_type_argument)

        create_image_file(open(input_argument, "r"), vocab_dict, target_sentence_ids, data_type_argument)

        create_single_oracle(open(input_argument, "r"), target_sentence_ids, data_type_argument)
    else:
        print("*** Invalid arguments provided.")


def create_titles_file(target_file, vocab_dict, data_type_argument):
    target_sentences_dict = {}
    print("*** Creating Title file...")
    title_file = open(os.path.dirname(os.path.realpath(target_file.name)) +
                      "/usp-" + data_type_argument + ".title", "w")

    for line_count, line in enumerate(target_file):
        title_uuid = "usp-" + uuid.uuid4().hex
        target_sentences_dict[title_uuid] = line
        word_id_sent = convert_sentence_to_word_ids(line, vocab_dict)
        title_file.write(title_uuid + "\n")
        title_file.write(" ".join(word_id_sent) + "\n\n")

    title_file.close()
    print("*** Finished Title file...")

    return target_sentences_dict


def create_image_file(input_file, vocab_dict, target_sentence_dict, data_type_argument):
    print("*** Creating Image file...")

    image_file = open(os.path.dirname(os.path.realpath(input_file.name)) +
                      "/usp-" + data_type_argument + ".image", "w")

    word_id_sent_list = []
    for line in input_file:
        word_id_sent = convert_sentence_to_word_ids(line, vocab_dict)
        word_id_sent_list.append(" ".join(word_id_sent) + "\n\n")

    target_sentence_ids = list(target_sentence_dict.keys())
    for title_count, title_id in enumerate(target_sentence_ids):
        image_file.write(title_id + "\n")
        if title_count <= (len(word_id_sent_list) - 1):
            image_file.write(word_id_sent_list[title_count])

    image_file.close()
    print("*** Finished Image file...")



def create_single_oracle(input_file, target_sentence_dict, data_type_argument):
    print("*** Fetching Universal Sentence Encoder embeddings")
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    print("*** Creating Single Oracle file...")

    oracle_file_path = os.path.dirname(os.path.realpath(input_file.name)) + \
                       "/usp-" + data_type_argument + ".label.singleoracle"
    logging_file_path = os.path.dirname(os.path.realpath(input_file.name)) + "/usp-" + data_type_argument + ".log.txt"


    print("*** Creating new Oracle and Logging Files...")
    oracle_file = open(oracle_file_path, "w")
    oracle_file.close()

    logging_file = open(logging_file_path, "w")
    logging_file.close()

    target_sentence_ids = list(target_sentence_dict.keys())

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        tf_placeholder = tf.placeholder(tf.string, shape=[None])

        for line_count, line in enumerate(input_file):
            oracle_file = open(oracle_file_path, "a")
            logging_file = open(logging_file_path, "a")

            if line_count <= (len(target_sentence_ids) - 1):
                title_uuid = target_sentence_ids[line_count]
                target_sentence = target_sentence_dict[title_uuid]

                oracle_file.write(title_uuid + "\n")

                print("*** Processing line number {}".format(line_count))
                sent_tokenize_list = sent_tokenize(line)
                print("*** Number of sentences: {}".format(len(sent_tokenize_list)))

                logging_file.write("*** Processing line number {} \n".format(line_count))
                logging_file.write("*** Number of sentences: {} \n".format(len(sent_tokenize_list)))


                distances = prepare_labels(embed, tf_placeholder, sent_tokenize_list, target_sentence)

                for dist_count, a_distance in enumerate(distances):
                    oracle_file.write(str(int(a_distance)) + "\n")
                    if dist_count == (len(distances) - 1):
                        oracle_file.write("\n")
            else:
                logging_file.write("*** Line number {} is greater than target sentences length {} \n".format(line_count,
                                                                                                             len(target_sentence_ids) - 1))
            oracle_file.close()
            logging_file.close()
    logging_file = open(logging_file_path, "w")
    logging_file.write("*** Finished creating Single Oracle file.")
    logging_file.close()


def prepare_labels(embed, tf_placeholder, source_sentences:list, target:str, topk:int=3):
    # function returning ids of topk number of best sentences
    target_embeddings = embed([target])
    target_embeds = target_embeddings.eval(feed_dict={tf_placeholder: [target]})
    target_vector = np.array(target_embeds)

    source_embeddings = embed(source_sentences)
    source_embeds = source_embeddings.eval(feed_dict={tf_placeholder:  source_sentences})
    source_matrix = np.array(source_embeds)

    cosine_distances = []
    for i, w in enumerate(source_matrix):
            # store simmilarity measures betwean each of the sentences and the target
            cosine_score = spatial.distance.cosine(target_vector, w)
            cosine_distances.append(cosine_score)
    cosine_distances = np.array(cosine_distances)
    lowest_cosine_distances = cosine_distances.argsort()[:topk]

    target_binarized = np.zeros(len(cosine_distances))
    np.put(target_binarized, lowest_cosine_distances, 1, mode='clip')

    return target_binarized.tolist()



def main():
    parse_arguments()


if __name__ == '__main__':
    main()
