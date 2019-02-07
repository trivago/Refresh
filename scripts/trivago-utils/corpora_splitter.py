# corpora_splitter.py -- Performs sentence segmentation and splits each sentence in a seperate file when called.
# @author Saad Mahamood

import argparse
import os
import os.path
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy import spatial


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='File to process', required=True)
    parser.add_argument('-o', '--output', help='Output directory', required=True)

    args = parser.parse_args()

    filename_argument = args.file
    output_argument = args.output

    if filename_argument and output_argument and os.path.isfile(filename_argument):
        print("*** Opening file...")
        file_obj = open(filename_argument, "r")
        output_directory_path = output_argument + "/" + Path(file_obj.name).stem
        print("*** Creating output directory with path: " + output_directory_path)
        os.mkdir(output_directory_path)
        split_file(file_obj, output_directory_path)

    else:
        print("*** Invalid arguments provided.")


def split_file(input_file, output_dir):
    print("*** Sentence segmenting and splitting file.")
    for line_count, line in enumerate(input_file):

        print("*** Processing line number {}".format(line_count))
        sent_tokenize_list = sent_tokenize(line)
        print("*** Number of sentences: {}".format(len(sent_tokenize_list)))
        for sentence_count, a_tokenzie_sent in enumerate(sent_tokenize_list):
            sentence_file_name = str(line_count) + "-" + str(sentence_count) + "-" + os.path.basename(input_file.name)

            word_tokenized_sent = word_tokenize(a_tokenzie_sent)
            print("*** Writing file : " + sentence_file_name)
            output_file = open(output_dir + "/" + sentence_file_name, "w")
            output_file.write(" ".join(word_tokenized_sent))
            output_file.close()

#function returning ids of topk number of best sentences
def prepare_labels(source_sentences:list, target:str, topk:int=3):
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
                #remember the target vector
                target = w
            else:
                #store simmilarity measures betwean each of the sentences and the target
                distances.append(spatial.distance.cosine(target, w))
        distances = np.array(distances)
        distances.argsort()[-topk:][::-1]
        return distances


def main():
    parse_arguments()


if __name__ == '__main__':
    main()
