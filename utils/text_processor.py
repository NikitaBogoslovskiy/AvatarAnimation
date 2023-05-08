import os
import codecs
import random
from nltk.tokenize import sent_tokenize
from config.paths import PROJECT_DIR


class TextProcessor:
    @staticmethod
    def find_sentences_with_particular_length(texts_directory, output_path, min_length, max_length, max_sentences_number):
        text_names = next(os.walk(texts_directory), (None, None, []))[2]
        processed = []
        for name in text_names:
            # f = codecs.open(texts_directory + name, "r", "utf-8")
            with open(texts_directory + name, "r", encoding="utf-8") as f:
                text = f.read().replace("\n", ". ")
                processed.extend(list(filter(lambda x: min_length <= len(x) <= max_length, sent_tokenize(text, language="russian"))))
            # f.close()
        random.shuffle(processed)
        processed = [f"{i + 1}) " + x + "\n\n" for i, x in enumerate(processed)]
        if len(processed) > max_sentences_number:
            processed = processed[:max_sentences_number]
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(processed)


if __name__ == "__main__":
    TextProcessor.find_sentences_with_particular_length(texts_directory=f"{PROJECT_DIR}/other_data/input_texts/",
                                                        output_path=f"{PROJECT_DIR}/other_data/output_sentences/sentences.txt",
                                                        min_length=60,
                                                        max_length=80,
                                                        max_sentences_number=2000)
