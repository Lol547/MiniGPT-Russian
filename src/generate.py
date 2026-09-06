import torch
import torch.nn.functional as F
from functools import partial


def greedy_search(logits):
    """Жадный поиск - выбор наиболее вероятного токена."""
    return torch.argmax(logits, dim=-1, keepdim=True)


def random_sample(logits, temperature=1.0):
    """Случайное сэмплирование с температурой."""
    if temperature <= 0:
        raise ValueError("Temperature должна быть больше 0")
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_k_sample(logits, k=5, temperature=1.0):
    """Top-K сэмплирование."""
    if temperature <= 0:
        raise ValueError("Temperature должна быть больше 0")
    if k <= 0:
        raise ValueError("k должна быть больше 0")
    
    k = min(k, logits.size(-1))
    logits = logits / temperature
    top_logits, top_indices = torch.topk(logits, k=k, dim=-1)
    probs = F.softmax(top_logits, dim=-1)
    choice_idx = torch.multinomial(probs, num_samples=1)
    return top_indices.gather(-1, choice_idx)


def top_p_sample(logits, p=0.9, temperature=1.0):
    """Top-P (Nucleus) сэмплирование."""
    if temperature <= 0:
        raise ValueError("Temperature должна быть больше 0")
    if not (0 < p <= 1):
        raise ValueError("p должна быть в диапазоне (0, 1]")
    
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum_probs <= p
    mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=-1)
    sorted_probs = sorted_probs * mask.float()
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_idx)


def top_k_top_p_sample(logits, k=50, p=0.9, temperature=1.0):
    """Комбинированное Top-K + Top-P сэмплирование."""
    if temperature <= 0:
        raise ValueError("Temperature должна быть больше 0")
    if k <= 0:
        raise ValueError("k должна быть больше 0")
    if not (0 < p <= 1):
        raise ValueError("p должна быть в диапазоне (0, 1]")
    
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k=k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum <= p
    mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=-1)
    sorted_probs = sorted_probs * mask.float()
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    final_idx = sorted_indices.gather(-1, sampled_idx)
    return top_k_indices.gather(-1, final_idx)


def generate_text(
    prompt,
    model,
    tokenizer,
    max_new_tokens=64,
    sample_fn=greedy_search,
    device='cuda',
    repetition_penalty=1.0,
    pad_id=0,
    bos_id=None,
    eos_id=None,
    sequence_length=256
):
    """
    Генерирует текст по заданному промпту.
    
    Args:
        prompt: начальный текст
        model: модель MiniGPT
        tokenizer: токенизатор
        max_new_tokens: максимальное число новых токенов(после 256 начинает забывать информацию)
        sample_fn: функция сэмплирования
        device: 'cuda' или 'cpu'
        repetition_penalty: штраф за повторения (>1.0 подавляет повторения)
        pad_id: ID паддинга
        bos_id: ID начала последовательности
        eos_id: ID конца последовательности
        sequence_length: максимальная длина последовательности
    
    Returns:
        сгенерированный текст
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if generated.size(1) > sequence_length:
                context = generated[:, -sequence_length:]
            else:
                context = generated

            logits = model(context)
            next_logits = logits[:, -1, :]

            if pad_id is not None:
                next_logits[:, pad_id] = -float("inf")
            if bos_id is not None:
                next_logits[:, bos_id] = -float("inf")

            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    next_logits[:, token_id] /= repetition_penalty

            next_token = sample_fn(next_logits)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output
