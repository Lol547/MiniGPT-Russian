import os
import random
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from model import MiniGPT
from dataset import StreamingTextDataset, CachedValDataset
from utils import NoamScheduler, get_optimizer_params


def main():
    parser = argparse.ArgumentParser(description="Обучение MiniGPT")
    parser.add_argument("--epochs", type=int, default=15, help="Количество эпох")
    parser.add_argument("--steps_per_epoch", type=int, default=10000, help="Шагов на эпоху")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер батча")
    parser.add_argument("--sequence_length", type=int, default=256, help="Длина последовательности")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Размер скрытого слоя")
    parser.add_argument("--intermediate_dim", type=int, default=2048, help="Размер FFN")
    parser.add_argument("--num_heads", type=int, default=8, help="Количество голов внимания")
    parser.add_argument("--num_layers", type=int, default=8, help="Количество слоёв")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")
    parser.add_argument("--val_steps", type=int, default=500, help="Шагов валидации")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Размер буфера для перемешивания")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
    
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "[PAD]"
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = "[EOS]"
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = "[BOS]"
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = "[UNK]"
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    
    PAD_ID = tokenizer.pad_token_id
    EOS_ID = tokenizer.eos_token_id
    BOS_ID = tokenizer.bos_token_id
    VOCAB_SIZE = len(tokenizer)
    
    print(f"Размер словаря: {VOCAB_SIZE}")
    print(f"PAD: {PAD_ID}, BOS: {BOS_ID}, EOS: {EOS_ID}")
    
    print("Загрузка датасета PotatoHD/ru-text-corpus...")
    dataset = load_dataset(
        "PotatoHD/ru-text-corpus",
        split="train",
        streaming=True,
        cache_dir="./dataset_cache"
    )
    
    TRAIN_DOCS = 200000
    VAL_DOCS = 10000
    train_iter = dataset.take(TRAIN_DOCS)
    val_iter = dataset.skip(TRAIN_DOCS).take(VAL_DOCS)
    
    train_stream = StreamingTextDataset(
        train_iter, tokenizer, args.sequence_length, EOS_ID,
        buffer_size=args.buffer_size, shuffle=True
    )
    train_loader = DataLoader(train_stream, batch_size=args.batch_size, pin_memory=True)
    
    num_val_samples = args.val_steps * args.batch_size
    val_cache = CachedValDataset(
        val_iter, tokenizer, args.sequence_length, EOS_ID, num_val_samples
    )
    val_loader = DataLoader(val_cache, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    print(f"Примерное число шагов на эпоху: {args.steps_per_epoch}, валидационных шагов: {args.val_steps}")
    
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        sequence_length=args.sequence_length,
        pad_id=PAD_ID,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        get_optimizer_params(model),
        lr=0.0,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    scheduler = NoamScheduler(optimizer, d_model=args.hidden_dim, warmup_steps=args.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.03, reduction="mean")
    scaler = torch.amp.GradScaler(device="cuda", enabled=torch.cuda.is_available())
    
    TOTAL_STEPS = args.epochs * args.steps_per_epoch
    VAL_EVERY = args.steps_per_epoch // 2
    
    best_val_loss = float("inf")
    train_iterator = iter(train_loader)
    
    total_train_loss = 0.0
    total_train_acc = 0
    total_train_tokens = 0
    
    global_pbar = tqdm(total=TOTAL_STEPS, desc="Обучение")
    model.train()
    
    for step in range(1, TOTAL_STEPS + 1):
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)
        
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        mask = (y != PAD_ID)
        num_tokens = mask.sum().item()
        total_train_loss += loss.item() * num_tokens
        total_train_tokens += num_tokens
        
        preds = torch.argmax(logits, dim=-1)
        correct = ((preds == y) & mask).sum().item()
        total_train_acc += correct
        
        current_loss = total_train_loss / total_train_tokens if total_train_tokens > 0 else 0.0
        current_acc = (total_train_acc / total_train_tokens * 100) if total_train_tokens > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        
        global_pbar.update(1)
        global_pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.1f}%",
            "lr": f"{current_lr:.6f}"
        })
        
        if step % VAL_EVERY == 0 or step == TOTAL_STEPS:
            avg_train_loss = total_train_loss / total_train_tokens if total_train_tokens > 0 else 0.0
            train_accuracy = (total_train_acc / total_train_tokens * 100) if total_train_tokens > 0 else 0.0
            
            model.eval()
            total_val_loss = 0.0
            total_val_acc = 0
            total_val_tokens = 0
            
            with torch.no_grad():
                for batch_idx, (x_val, y_val) in enumerate(val_loader):
                    if batch_idx >= args.val_steps:
                        break
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits_val = model(x_val)
                        loss_val = criterion(logits_val.reshape(-1, VOCAB_SIZE), y_val.reshape(-1))
                    
                    mask_val = (y_val != PAD_ID)
                    num_tokens_val = mask_val.sum().item()
                    total_val_loss += loss_val.item() * num_tokens_val
                    total_val_tokens += num_tokens_val
                    
                    preds_val = torch.argmax(logits_val, dim=-1)
                    correct_val = ((preds_val == y_val) & mask_val).sum().item()
                    total_val_acc += correct_val
            
            avg_val_loss = total_val_loss / total_val_tokens if total_val_tokens > 0 else 0.0
            val_accuracy = (total_val_acc / total_val_tokens * 100) if total_val_tokens > 0 else 0.0
            val_ppl = math.exp(min(avg_val_loss, 20))
            
            print(f"Шаг {step}/{TOTAL_STEPS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val PPL: {val_ppl:.2f} | LR: {current_lr:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "sequence_length": args.sequence_length,
                        "pad_id": PAD_ID,
                        "hidden_dim": args.hidden_dim,
                        "intermediate_dim": args.intermediate_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                    }
                }
                torch.save(checkpoint, "best_mini_gpt.pt")
                print(f"Чекпоинт сохранён (Val Loss: {avg_val_loss:.4f})")
            
            total_train_loss = 0.0
            total_train_acc = 0
            total_train_tokens = 0
            model.train()
    
    global_pbar.close()
    
    tokenizer.save_pretrained("./tokenizer/")
    print("Обучение завершено. Токенизатор сохранён.")


if __name__ == "__main__":
    main()
