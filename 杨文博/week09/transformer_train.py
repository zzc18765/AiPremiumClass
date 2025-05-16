PAD_TOKEN_ID = 3

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        # 生成 mask: [batch_size, seq_len]
        padding_mask = (input_ids == PAD_TOKEN_ID).bool()

        # 前向传播
        logits = model(input_ids, src_key_padding_mask=padding_mask)

        # [batch * seq_len, vocab_size] vs [batch * seq_len]
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
