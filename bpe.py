import xml.etree.ElementTree as ET


def bpe_decomp(train_data, merge_operations):
    learned_rules = []
    tree = ET.parse(train_data)
    root = tree.getroot()

    all_words = []

    # fill all_words with lists of words
    for sent in root.findall('.//orth'):
        for i in sent.text.split():
            letters = list(i)
            letters[-1] += '</w>'
            all_words.append(letters)

    # merge operations
    for _ in range(merge_operations):
        pairs_dict = {}

        # count pairs
        for word in all_words:
            for j in range(len(word)-1):
                pair = word[j] + word[j+1]
                pairs_dict[pair] = pairs_dict.get(pair, 0) + 1
        if not pairs_dict:
            break
        # get max pair and add to rules 
        top_pair = max(pairs_dict, key=pairs_dict.get)
        learned_rules.append(top_pair)
        new_all_letters = []

        # apply new rule
        for word in all_words:

            # fast check if rule is in current word
            if not any(word[k] + word[k+1] == top_pair for k in range(len(word)-1)):
                new_all_letters.append(word)
                continue

            # loop through words
            new_letters_list = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] + word[i+1] == top_pair:
                    new_letters_list.append(top_pair)
                    i += 2
                else:
                    new_letters_list.append(word[i])
                    i += 1
            new_all_letters.append(new_letters_list)
        all_words = new_all_letters
    return learned_rules
def bpe_apply(text, learned_rules):
    words = []
    for w in text.split():
        letters = list(w)
        letters[-1] += '</w>'
        words.append(letters)

    for rule in learned_rules:
        new_words = []
        for word in words:

            # fast check if rule is in current word
            if not any(word[k] + word[k+1] == rule for k in range(len(word)-1)):
                new_words.append(word)
                continue
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] + word[i+1] == rule:
                    new_word.append(rule)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        words = new_words

    return words

def bpe_reverse(data):
    flat_tokens = [token for word in data for token in word]
    text = "".join(flat_tokens)
    text = text.replace('</w>', ' ')
    return text.strip()

def test_merges(train_data, merges : list[int]):

    print("bpe_decomp...")
    rules = bpe_decomp(train_data, max(merges))
    print("bpe_decomp done!")

    # test save rules and load rules
    save_rules(rules, "ResultsA2/Rules.txt")
    rules = load_rules("ResultsA2/Rules.txt")

    # get sentences from path
    tree = ET.parse(train_data)
    sentences = [s.text for s in tree.getroot().findall(".//orth")]

    # calc different vocabularies
    print("calc vocabularies...")
    for merge in merges:
        voc = set()
        for s in sentences:
            for word in bpe_apply(s, rules[:merge]):
                for token in word:
                    voc.add(token)
        print(f"Merges: {merge} with voc length: {len(voc)}")

# save rules
def save_rules(rules, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rules:
            f.write(r + "\n")

# load rules
def load_rules(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


if __name__ == "__main__":
    test_merges("train_100_corpus.xml", [25, 50, 100, 200])
    # results:
    # Merges: 25 with voc length: 79
    # Merges: 50 with voc length: 104
    # Merges: 100 with voc length: 154
    # Merges: 200 with voc length: 254