import random
import torch
from torch.utils.data import IterableDataset, Dataset


class StreamingTextDataset(IterableDataset):
    """Потоковый датасет, обрабатывающий текст по частям."""
    
    def __init__(
        self,
        dataset,
        tokenizer,
        seq_len,
        eos_id,
        pad_id=None,
        buffer_size=10000,
        shuffle=True,
        max_doc_tokens=2048
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.max_doc_tokens = max_doc_tokens

    def __iter__(self):
        buffer = []
        window_buffer = [] if self.shuffle else None

        for example in self.dataset:
            tokens = self.tokenizer(
                example["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_doc_tokens
            )["input_ids"]
            tokens.append(self.eos_id)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                window = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]

                if self.shuffle:
                    window_buffer.append(window)
                    if len(window_buffer) >= self.buffer_size:
                        idx = random.randint(0, len(window_buffer) - 1)
                        w = window_buffer.pop(idx)
                        yield (
                            torch.tensor(w[:-1], dtype=torch.long),
                            torch.tensor(w[1:], dtype=torch.long)
                        )
                else:
                    yield (
                        torch.tensor(window[:-1], dtype=torch.long),
                        torch.tensor(window[1:], dtype=torch.long)
                    )

        if self.shuffle and window_buffer:
            random.shuffle(window_buffer)
            for w in window_buffer:
                yield (
                    torch.tensor(w[:-1], dtype=torch.long),
                    torch.tensor(w[1:], dtype=torch.long)
                )


class CachedValDataset(Dataset):
    """Кэшированный датасет для валидации."""
    
    def __init__(self, dataset_iter, tokenizer, seq_len, eos_id, num_samples):
        self.samples = []
        buffer = []
        
        for example in dataset_iter:
            tokens = tokenizer(
                example["text"],
                add_special_tokens=False,
                truncation=True,
                max_length=2048
            )["input_ids"]
            tokens.append(eos_id)
            buffer.extend(tokens)
            
            while len(buffer) >= seq_len + 1 and len(self.samples) < num_samples:
                window = buffer[:seq_len + 1]
                buffer = buffer[seq_len:]
                self.samples.append((
                    torch.tensor(window[:-1], dtype=torch.long),
                    torch.tensor(window[1:], dtype=torch.long)
                ))
            if len(self.samples) >= num_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
