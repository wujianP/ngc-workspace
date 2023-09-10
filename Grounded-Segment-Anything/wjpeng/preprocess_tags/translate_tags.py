from googletrans import Translator
import argparse


def translate_word(word):
    translator = Translator()
    translations = translator.translate(word, dest='zh-cn')
    return translations


def save_translation_to_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file, 'w', encoding='utf-8') as file:

        for idx, line in enumerate(lines):
            english_word = line.strip()
            translations = translate_word(english_word)
            chinese_meanings = translations.text

            line_to_write = f"{idx},{english_word},{chinese_meanings}\n"
            print(line_to_write)
            file.write(line_to_write)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--dest_file', type=str)
    args = parser.parse_args()

    save_translation_to_file(args.src_file, args.dest_file)
