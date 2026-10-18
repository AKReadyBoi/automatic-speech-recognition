import json
from typing import List, Union, Iterable

class Dictionary:

    # initializes with [blank] = 0 and [unk] = 1
    def __init__(self):
        self._str2idx = {}
        self._idx2str = {}
        
        self.blank_token = "[blank]"
        self.unk_token = "[unk]"
        
        self.add_token(self.blank_token)
        self.add_token(self.unk_token)


    # returns length of dictionary
    def __len__(self):
        return len(self._str2idx)

    # handle blank index
    @property
    def blank_idx(self):
        return self._str2idx[self.blank_token]

    # adds token to dictionary, returns the tokens index
    def add_token(self, token: str) -> int:
        if token not in self._str2idx:
            idx = len(self._str2idx)
            self._str2idx[token] = idx
            self._idx2str[idx] = token
        return self._str2idx[token]



    # transforms a token or a list of tokens into indices
    def encode(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:

        unk_idx = self._str2idx[self.unk_token]
        
        if isinstance(tokens, str):
            return self._str2idx.get(tokens, unk_idx)
        else:
            num_list = []
            for token in tokens:
                found = self._str2idx.get(token, unk_idx)
                num_list.append(found)

            return num_list



    # transforms an index or a list of indices back into tokens
    def decode(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:

        unk_token = self.unk_token
        
        if isinstance(indices, int):
            return self._idx2str.get(indices, unk_token)
        else:
            str_list = []
            for idx in indices:
                found = self._idx2str.get(idx, unk_token)
                str_list.append(found)
            return str_list



    # saves dict as .json
    def save(self, filepath: str):

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._str2idx, f, ensure_ascii=False, indent=2)


    # loads dict from .json
    def load(self, filepath: str):

        with open(filepath, 'r', encoding='utf-8') as f:
            self._str2idx = json.load(f)
            # recreate bidirectional relation
            self._idx2str = {int(v): k for k, v in self._str2idx.items()}



    # creates dict from bpe preprocessed sentences
    @classmethod
    def build_from_bpe_text(cls, bpe_sentences: Iterable[List[str]]) -> 'Dictionary':

        vocab = cls()
        for sentence in bpe_sentences:
            for token in sentence:
                vocab.add_token(token)
        return vocab


