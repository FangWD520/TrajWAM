# Model download

The complete inference checkpoint is packaged as `TrajWAM-v1.0.0.zip`.
Its Baidu Netdisk download link will be added here after upload and verification.
The checkpoint files are not stored in this Git repository.

After downloading:

```bash
unzip TrajWAM-v1.0.0.zip
cd TrajWAM-v1.0.0
sha256sum -c SHA256SUMS
```

Supply the absolute path to the extracted directory to the evaluation script's
`--checkpoint` option. Keep both safetensors shards, their index, and every
configuration, tokenizer and processor file together. Optimizer states are not
needed. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup and
closed-loop evaluation.

The package includes the model licenses and attribution. See
[MODEL_CARD.md](MODEL_CARD.md) for the checkpoint and intended use.
