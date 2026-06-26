"""Minimal training demo using the Python wrapper."""

from aurelis import Trainer

DATA = "data/simple_language.txt"

config = {
    "vocab_size": 64,
    "D": 64,
    "d_model": 32,
    "d_tau": 32,
    "d_ff": 256,
    "num_layers": 2,
    "lr": 0.01,
}

trainer = Trainer(config)
print(f"Parameters: {trainer.num_params:,}")

trainer.load_data(DATA)
print(f"Vocab size: {trainer.vocab_size}")

for epoch in range(1, 4):
    loss = trainer.train_epoch()
    acc = trainer.accuracy
    print(f"epoch {epoch}: loss={loss:.4f}  acc={acc:.3f}")

trainer.close()
