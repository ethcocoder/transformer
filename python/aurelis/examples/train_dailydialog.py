"""Train on DailyDialog dataset using the Aurelis LENS model with BPE tokenizer."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from aurelis import Trainer

# Config for conversational AI
config = {
    "vocab_size": 4096,
    "D": 256,
    "d_model": 128,
    "d_tau": 128,
    "d_ff": 1024,
    "num_layers": 6,
    "lr": 0.001,
    "lambda_aux": 0.001,
    "lambda_stab": 0.01,
}

trainer = Trainer(config)
print(f"Parameters: {trainer.num_params:,}")
print(f"Vocab target: {config['vocab_size']}")

# Load DailyDialog (preprocessed by scripts/preprocess_dailydialog.py)
data_path = "data/dailydialog.txt"
trainer.load_data(data_path)
print(f"Vocab size: {trainer.vocab_size}")

# Train for several epochs
for epoch in range(1, 11):
    loss = trainer.train_epoch()
    acc = trainer.accuracy
    print(f"epoch {epoch:2d}: loss={loss:.4f}  acc={acc:.3f}")

trainer.close()
