from transformers import PreTrainedTokenizerFast
from collections import Counter
from dataclasses import dataclass
import numpy as np


@dataclass
class RougeScoreResult:
    precision: float
    recall: float
    fmeasure: float

    @classmethod
    def build(cls, precision: float, recall: float):
        if precision + recall <= 0:
            return cls(precision, recall, 0)

        fmeasure = 2 * (precision * recall) / (precision + recall)

        return cls(precision, recall, fmeasure)


class RougeScore:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        rouge_n: list[int] = [1],
        rouge_l: bool = False,
    ):
        self._rouge_n = rouge_n
        self._rouge_l = rouge_l
        self._tokenizer = tokenizer

    def score(self, prediction: str, target: str) -> dict[str, RougeScoreResult]:
        target_tokens = self._tokenizer.tokenize(target)
        prediction_tokens = self._tokenizer.tokenize(prediction)
        result = {}

        if self._rouge_l:
            result["rougeL"] = RougeScore._score_lcs(target_tokens, prediction_tokens)

        for n in self._rouge_n:
            target_n_grams = RougeScore._build_n_gram(target_tokens, n)
            prediction_n_grams = RougeScore._build_n_gram(prediction_tokens, n)
            scores = RougeScore._score_n_grams(target_n_grams, prediction_n_grams)
            result[f"rouge{n}"] = scores

        return result

    def avg_score(
        self, predictions: list[str], targets: list[str]
    ) -> dict[str, RougeScoreResult]:

        all_result = [
            self.score(prediction, target)
            for prediction, target in zip(predictions, targets)
        ]
        result = dict()

        result["avg rougeL"] = np.mean([item["rougeL"].fmeasure for item in all_result])

        for n in self._rouge_n:
            result[f"avg rouge{n}"] = np.mean(
                [item[f"rouge{n}"].fmeasure for item in all_result]
            )

        return result

    @staticmethod
    def build_score(precision: float, recall: float):
        return RougeScoreResult.build(precision, recall)

    @staticmethod
    def _build_n_gram(tokens: list[str], n: int) -> dict:
        # n_grams = Counter()
        n_grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        n_grams = Counter(n_grams)
        return n_grams

    @staticmethod
    def _score_lcs(target_tokens: list[str], prediction_tokens: list[str]):

        if not target_tokens or not prediction_tokens:
            return RougeScore.build_score(precision=0, recall=0)

        # Compute length of LCS from the bottom up in a table (DP approach).
        lcs_table = RougeScore._lcs_table(target_tokens, prediction_tokens)
        lcs_length = lcs_table[-1][-1]

        precision = lcs_length / len(prediction_tokens)
        recall = lcs_length / len(target_tokens)

        return RougeScore.build_score(precision=precision, recall=recall)

    @staticmethod
    def _lcs_table(ref: list[str], can: list[str]):
        """Create 2-d LCS score table."""
        rows, cols = len(ref), len(can)
        lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if ref[i - 1] == can[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
        return lcs_table

    @staticmethod
    def _score_n_grams(
        target_n_grams: Counter[tuple[str, ...]],
        prediction_n_grams: Counter[tuple[str, ...]],
    ):
        intersection_ngrams_count = sum(
            min(target_n_grams[ngram], prediction_n_grams.get(ngram, 0))
            for ngram in target_n_grams
        )

        # Calculate total counts for precision and recall
        target_ngrams_count = sum(target_n_grams.values())
        prediction_ngrams_count = sum(prediction_n_grams.values())

        # Calculate precision, recall, and F1
        precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
        recall = intersection_ngrams_count / max(target_ngrams_count, 1)

        return RougeScore.build_score(precision, recall)
