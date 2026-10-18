import soundfile
import os
import xml.etree.ElementTree as ET


def corpus_statistics_audio():
    length = []
    for i in os.listdir("audio"):
        full_path = os.path.join("audio", i)
        length.append(soundfile.info(file=full_path).duration)
    print("maximum= ", max(length))
    print("minimum= ", min(length))
    print("sum= ", sum(length))
    print("amount= ", len(length))
    print("average= ", sum(length) / len(length))


def corpus_statistics_text():
    tree = ET.parse("train_100_corpus.xml")
    root = tree.getroot()

    word_length = []
    global_vocab = set()

    for i in root.findall('.//orth'):
        words = i.text.split()
        word_length.append(len(words))
        global_vocab.update(words)

    print("sequences= ", len(word_length))
    print("words with repeat= ", sum(word_length))
    print("different words= ", len(global_vocab))
    print("average= ", sum(word_length) / len(word_length))


corpus_statistics_text()
corpus_statistics_audio()
