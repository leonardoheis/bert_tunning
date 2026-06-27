import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase


def prepare_text(text: str, tokenizer: PreTrainedTokenizerBase, strategy: str = "first") -> str:
    if strategy == "first":
        return text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if strategy == "middle":
        max_t = tokenizer.model_max_length
        if len(tokens) <= max_t - 2:
            return text
        half = (max_t - 2) // 2
        selected = tokens[:half] + tokens[-half:]
        return str(tokenizer.decode(selected, skip_special_tokens=True))
    return text


class ClassiflowDataset(TorchDataset):  # type: ignore[type-arg]
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
