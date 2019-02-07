# vocab_mapper.py -- Maps vocab words or punctuation onto word IDs when called or vice-versa.
# @author Saad Mahamood

import argparse
import os
import os.path
from nltk.tokenize import word_tokenize
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', help='Vocab File of Strings', required=True)
    parser.add_argument('-o', '--output', help='Model output file', required=False)
    parser.add_argument('-i', '--input_dir', help='Model output file', required=False)
    parser.add_argument('-m', '--mode', help='Conversion Mode: \'output\' for converting a single output file of '
                                             'word IDs to words '
                                             'or \'dir\' for converting a directory of copora to word IDs', required=True)

    args = parser.parse_args()

    vocab_argument = args.vocab
    output_argument = args.output
    input_dir_argument = args.input_dir
    mode_argument = args.mode

    if output_argument or input_dir_argument:
        if os.path.isfile(vocab_argument):
            vocab_file = open(vocab_argument, "r")

            if output_argument and mode_argument == 'output':
                if os.path.isfile(output_argument):
                    print("*** Processing output file...")
                    output_file = open(output_argument, "r")

                    process_output_file(vocab_file, output_file)
            elif input_dir_argument and mode_argument == 'dir':
                if os.path.isdir(input_dir_argument):
                    print("*** Processing input directory...")
                    if not os.path.isdir(input_dir_argument + "/converted"):
                        os.mkdir(input_dir_argument + "/converted")
                    process_input_directory(vocab_file, input_dir_argument)
    else:
        print("*** Please supply output file or an input directory parameter.")


def process_output_file(vocab_file, output_file):
    vocab_list = vocab_file.read().splitlines()

    vocalised_file = open(os.path.dirname(os.path.realpath(output_file.name)) + "/vocab-output.txt", "w")
    for line_count, line in enumerate(output_file):
        print("*** Processing line number {}".format(line_count))

        # Replace word IDs with
        line_tokens = line.split(" ")
        vocalised_sentence = []
        if line_tokens and line_tokens[0] != '\n':
            for a_line_token in line_tokens:
                word = vocab_list[int(a_line_token)]
                vocalised_sentence.append(word)

            # Write output to file:
            vocalised_file.write(" ".join(vocalised_sentence) + "\n")
    vocalised_file.close()


def process_input_directory(vocab_file, input_dir):
    print("*** Building vocab dictionary")
    # Build vocab dictionary
    vocab_dict = {}
    for line_count, line in enumerate(vocab_file):
        vocab_dict[str(line.rstrip("\n"))] = line_count

    print("*** Vocabulary size: {}".format(len(vocab_dict)))

    # Get list of files in the directory:
    directory_list_contents = os.listdir(input_dir)
    if directory_list_contents:
        for directory_item in directory_list_contents:
            if os.path.isfile(input_dir + "/" + directory_item):
                current_file = open(input_dir + "/" + directory_item, "r")
                if ".txt" == Path(current_file.name).suffix:
                    convert_file_to_word_ids(input_dir, vocab_dict, current_file)


def convert_file_to_word_ids(input_dir, vocab_dict, input_file):
    print("*** Converting {} from words to word IDs.".format(os.path.basename(input_file.name)))
    sentence_file_name = "word-id-" + os.path.basename(input_file.name)
    output_file = open(input_dir + "/converted/" + sentence_file_name, "w")
    for line_count, line in enumerate(input_file):

        word_tokenized_sent = word_tokenize(line.decode("utf-8"))

        word_id_sent = []
        for word_token in word_tokenized_sent:
            if word_token in vocab_dict:
                word_id_value = vocab_dict[word_token]
                word_id_sent.append(str(word_id_value))
            else:
                word_id_sent.append(str(1))

        print("*** Writing file : " + sentence_file_name)
        output_file.write(" ".join(word_id_sent) + "\n\n")
    output_file.close()



def main():
    parse_arguments()


if __name__ == '__main__':
    main()
