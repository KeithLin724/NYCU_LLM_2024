from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lightning as L
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
import re


@dataclass
class Lib:
    word2idx: dict[str, int]
    idx2word: dict[int, str]
    size: int

    def word_2_index(self, word: str) -> int:
        return self.word2idx.get(word, 0)

    def index_2_word(self, index: int) -> str:
        return self.idx2word.get(index, "")

    @staticmethod
    def natural_split(text: str) -> list[str]:
        pattern = r"([a-zA-Z0-9]+|[^\s\w]|[\s]+)"
        result = re.findall(pattern, text)
        return result

    @classmethod
    def build_from_text(cls, str_list: list[str]):
        group = " ".join(str_list)
        vocab = set(Lib.natural_split(group))

        word2idx = {word: i for i, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}

        return cls(word2idx, idx2word, len(word2idx))

    @classmethod
    def build_from_file(cls, filename: str):
        with open(filename, "r") as f:
            data = [item.strip() for item in f.readlines()]
        return cls.build_from_text(data)

    def sentence_to_idx_tensor(self, sentence: str):
        return torch.tensor(
            [self.word_2_index(word) for word in sentence.split(" ")], dtype=torch.long
        )

    def word_to_tensor(self, word: str) -> torch.Tensor:
        tensor = torch.zeros(self.size)
        index = self.word_2_index(word)

        tensor[index] = 1
        return tensor

    def tensor_to_sentence(self, tensor: torch.Tensor) -> str:
        return " ".join([self.index_2_word(int(item)) for item in tensor])

    def __len__(self) -> int:
        return self.size


class TextDataset(Dataset):
    def __init__(self, sentences: str, n: int, text_lib: Lib):
        self.data = []
        self.n = n
        self.text_lib = text_lib

        for sentence in tqdm(
            sentences, desc="process", unit="line", dynamic_ncols=True, leave=True
        ):
            line = sentence.split(" ")

            for i in range(len(line) - n):
                input_str = line[i : i + n]
                target_str = line[i + n]

                self.data.append(tuple([" ".join(input_str), target_str]))

        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_str, target_str = self.data[index]
        input_tensor = self.text_lib.sentence_to_idx_tensor(input_str)
        target_tensor = self.text_lib.word_to_tensor(target_str)
        return input_tensor, target_tensor

    @classmethod
    def build_from_file(cls, filename: str, n: int, lib: Lib = None):
        with open(file=filename, mode="r") as f:
            data = [line.strip() for line in f.readlines()]

        if lib is None:
            lib = Lib.build_from_text(data)

        return cls(data, n, lib)


class RnnModel(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_size: int,
        num_layers=1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        return

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        out = F.softmax(out, dim=1)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        input_tensor, target_tensor = train_batch

        output = self.forward(input_tensor)
        loss = F.cross_entropy(input=output, target=target_tensor)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        predicted_tensor, target_tensor_idx = torch.argmax(output, dim=1), torch.argmax(
            target_tensor, dim=1
        )
        accuracy = (predicted_tensor == target_tensor_idx).float().mean()

        self.log(
            "tran_acc",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):

        input_tensor, target_tensor = val_batch

        output = self.forward(input_tensor)
        loss = F.cross_entropy(input=output, target=target_tensor)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        predicted_tensor, target_tensor_idx = torch.argmax(output, dim=1), torch.argmax(
            target_tensor, dim=1
        )
        accuracy = (predicted_tensor == target_tensor_idx).float().mean()

        self.log(
            "test_acc",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss


epoch = 10
batch_size = 32
number_of_layer = 2
hidden = 128
embedding_dim = 128


lib = Lib.build_from_file("./train.txt")


model = RnnModel(
    vocab_size=lib.size,
    embedding_dim=embedding_dim,
    hidden_size=hidden,
    output_size=lib.size,
    num_layers=number_of_layer,
)


print(model)


tb_logger = pl_loggers.TensorBoardLogger("logs/")

checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="checkpoints",
    filename="model-{epoch:02d}-{train_loss:.2f}",
    save_top_k=3,
    mode="min",
)

trainer = L.Trainer(
    callbacks=[checkpoint_callback],
    accelerator="gpu",
    logger=tb_logger,
    default_root_dir="out/",
    max_epochs=epoch,
    # progress_bar_refresh_rate=1
)

train_dataset = TextDataset.build_from_file("./train.txt", n=2, lib=lib)
test_dataset = TextDataset.build_from_file("./test.txt", n=2, lib=lib)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

trainer.fit(model, train_dataloader, test_dataloader, ckpt_path="last")
