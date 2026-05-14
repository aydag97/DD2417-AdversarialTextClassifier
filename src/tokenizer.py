import re
import json
from collections import Counter, defaultdict


class TinyStoriesTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = []
        self.ids = {} 
               


    def _pretokenize(self, text:str) -> list[str]:
        regexp = r"\s*(?:\w+(?:'\w+)*|[^\w\s])"  # REPLACE WITH YOUR REGULAR EXPRESSION
        tokens = re.findall(regexp, text)
        return tokens


    def _generate_word_frequencies(self, words:list[str]) -> dict[tuple[str], int]:
        ret_dict = {}
        n = len(words)
        for i in range(n):
            splitted = tuple(re.findall(r".",words[i]))
            if splitted not in ret_dict:
                ret_dict[splitted] = 1
            else:
                ret_dict[splitted] += 1

        return ret_dict  


    def _count_token_bigrams(self, word_freqs: dict) -> defaultdict(int):
        bigram_counts = defaultdict(int)
        for token, freq in word_freqs.items():
            for i in range(len(token)-1):
                pair_token = (token[i], token[i+1])
                bigram_counts[pair_token] += freq

        return bigram_counts


    def _find_most_frequent_token_bigram(self, word_freqs:dict) -> tuple[str, str]:
        bigram_counts = self._count_token_bigrams(word_freqs)
        # If there are several most frequent bigrams, return the first one

        best_bigram = max(bigram_counts, key=bigram_counts.get)  

        return best_bigram


    def _merge_bigram(self, word_freqs: dict, best_bigram: tuple, new_token: str) -> dict[str, int]:
        new_word_freqs = {}
        for word_tuple, freq in word_freqs.items():
            new_word = []
            n = len(word_tuple)
            i = 0
            while i < n:
                if i != n-1 and (word_tuple[i], word_tuple[i+1]) == best_bigram:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_tuple = tuple(new_word)

            if new_word_tuple in new_word_freqs:
                new_word_freqs[new_word_tuple] += freq
            else:
                new_word_freqs[new_word_tuple] = freq

        return new_word_freqs


    def train(self, corpus_path:str):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        words = self._pretokenize(raw_text)
        word_freqs = self._generate_word_frequencies(words)

        # Initialize vocabulary with all unique individual characters found
        unique_chars = set()
        for word_tuple in word_freqs:
            for char in word_tuple:
                unique_chars.add(char)
        self.vocab = list(unique_chars)

        # Merging loop
        merges = {}  # (bigram) -> priority (lower number -> higher priority)
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            best_bigram = self._find_most_frequent_token_bigram(word_freqs)   
            self.merges[best_bigram] = i # Rank         
            new_token = "".join(best_bigram)
            self.vocab.append(new_token)
            word_freqs = self._merge_bigram(word_freqs, best_bigram, new_token)
            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_bigram} -> {new_token}")
        print(f"Merge {i+1}/{num_merges}: {best_bigram} -> {new_token}")
        self.ids = {v: k for k, v in enumerate(self.vocab)}


    def tokenize(self, text):
        tokens = []
        pretokens = self._pretokenize(text)
        for word in pretokens:
            parts = list(word)
            while len(parts) > 1:
                candidates = []

                for i in range(len(parts)-1):
                    pair = (parts[i], parts[i+1])
                    if pair in self.merges:
                        candidates.append((self.merges[pair],i,pair))

                if len(candidates) == 0:
                    break

                best = min(candidates)
                parts[best[1]:best[1]+2] = ["".join(best[2])]

            tokens.extend(parts)
        # changin this to handle the unknown tokens and setting their IDs to 0
        return tokens, [self.ids.get(t, 0) for t in tokens]


    def save(self, path):
        serializable_merges = {f"{k[0]}<SPLIT>{k[1]}": v for k, v in self.merges.items()}
        data = {"vocab": self.vocab, "merges": serializable_merges, "vocab_size": self.vocab_size, "ids": self.ids}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(vocab_size=data["vocab_size"])
        instance.vocab = data["vocab"]
        instance.ids = data["ids"]
        instance.merges = {tuple(k.split("<SPLIT>")): v for k, v in data["merges"].items()}
        return instance
