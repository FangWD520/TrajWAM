# Model download

Download `TrajWAM-v1.0.0.zip` from [Baidu Netdisk](https://pan.baidu.com/s/10GRbfS7DTxYw7yINi5rpQw?pwd=1234).

**Baidu Netdisk access code: `1234`.**
The checkpoint files are hosted separately from this Git repository.

After downloading and extracting the files, verify the two model shards:

```bash
cd /absolute/path/to/TrajWAM-v1.0.0
sha256sum -c /absolute/path/to/TrajWAM/docs/WEIGHT_SHARDS_SHA256SUMS
```

Supply the absolute path to the extracted directory to the evaluation script's
`--checkpoint` option. Keep both safetensors shards, their index, and every
configuration, tokenizer and processor file together. Optimizer states are not
needed. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup and
closed-loop evaluation.

The shard checksums apply to the model tensor files. Keep the configuration,
tokenizer and processor files supplied with this checkpoint; the loader resolves
the DINO constructor configuration using the configuration bundled with the code.

Model licenses and attribution are provided in [licenses/](../licenses/) and
[NOTICE](../NOTICE). See [MODEL_CARD.md](MODEL_CARD.md) for the checkpoint and
intended use.
