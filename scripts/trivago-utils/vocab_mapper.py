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
    parser.add_argument('-t', '--title_file', help="Tile file for inserting title UUIDS before each sentence.", required=False)

    args = parser.parse_args()

    vocab_argument = args.vocab
    output_argument = args.output
    input_dir_argument = args.input_dir
    mode_argument = args.mode
    title_file = args.title_file

    title_ids = process_title_file(title_file)

    if output_argument or input_dir_argument:
        if os.path.isfile(vocab_argument):
            vocab_file = open(vocab_argument, "r", encoding="utf-8")

            if output_argument and mode_argument == 'output':
                if os.path.isfile(output_argument):
                    print("*** Processing output file...")
                    output_file = open(output_argument, "r")

                    process_output_file(vocab_file, output_file, title_ids)
            elif input_dir_argument and mode_argument == 'dir':
                if os.path.isdir(input_dir_argument):
                    print("*** Processing input directory...")
                    if not os.path.isdir(input_dir_argument + "/converted"):
                        os.mkdir(input_dir_argument + "/converted")
                    process_input_directory(vocab_file, input_dir_argument)
        else:
            print("*** Vocab argument is not a file.")
    else:
        print("*** Please supply output file or an input directory parameter.")


def process_output_file(vocab_file, output_file, title_ids):
    # Build vocab dictionary
    vocab_dict = build_vocab_dict(vocab_file)
    convert_file_to_word_ids(os.path.dirname(os.path.realpath(output_file.name)), vocab_dict, output_file, title_ids)


def process_input_directory(vocab_file, input_dir):
    print("*** Building vocab dictionary")
    # Build vocab dictionary
    vocab_dict = build_vocab_dict(vocab_file)

    # Get list of files in the directory:
    directory_list_contents = os.listdir(input_dir)
    if directory_list_contents:
        for directory_item in directory_list_contents:
            if os.path.isfile(input_dir + "/" + directory_item):
                current_file = open(input_dir + "/" + directory_item, "r", encoding="utf-8")
                if ".txt" == Path(current_file.name).suffix:
                    convert_file_to_word_ids(input_dir, vocab_dict, current_file, [])


def convert_file_to_word_ids(input_dir, vocab_dict, input_file, title_ids):
    print("*** Converting {} from words to word IDs.".format(os.path.basename(input_file.name)))
    sentence_file_name = "word-id-" + os.path.basename(input_file.name)

    if not os.path.exists(input_dir + "/converted/"):
        os.makedirs(input_dir + "/converted/")

    output_file = open(input_dir + "/converted/" + sentence_file_name, "w")

    if title_ids:
        print("*** Converting with Title UUIDs")
        input_lines = input_file.readlines()
        for title_count, title in enumerate(title_ids):
            if title_count <= len(input_lines) - 1:
                output_file.write(title)
                convert_text_to_vocab(input_lines[title_count], output_file, vocab_dict)

    else:
        for line_count, line in enumerate(input_file):
            convert_text_to_vocab(line, output_file, vocab_dict)

    output_file.close()

def build_vocab_dict(vocab_file):
    vocab_dict = {}
    for line_count, line in enumerate(vocab_file):
        vocab_dict[str(line.rstrip("\n"))] = line_count

    print("*** Vocabulary size: {}".format(len(vocab_dict)))
    return vocab_dict


def process_title_file(title):
    title_list = []

    with open(title) as title_file:
        file_lines = title_file.readlines()

        for line in file_lines:
            if line.startswith("usp"):
                title_list.append(line)

    return title_list

def convert_text_to_vocab(text_line, output_file, vocab_dict):
    word_tokenized_sent = word_tokenize(text_line)
    word_id_sent = []
    for word_token in word_tokenized_sent:
        if word_token in vocab_dict:
            word_id_value = vocab_dict[word_token]
            word_id_sent.append(str(word_id_value))
        else:
            word_id_sent.append(str(1))

    output_file.write(" ".join(word_id_sent) + "\n\n")


def main():
    parse_arguments()


if __name__ == '__main__':
    main()
