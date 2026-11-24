"""Convert Sunbird's FP16 CTranslate2 checkpoint to MLX without downloading it again.

Layout references: CTranslate2 4.9 specs/model_spec.py::_serialize and
converters/transformers.py::WhisperLoader, and mlx_whisper.whisper.
Only binary v6 WhisperSpec revision 3, unquantized weights are accepted.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten
from mlx_whisper.whisper import ModelDimensions, Whisper

REPO = "Sunbird/faster-whisper-51-african-languages"
REVISION = "fcb165d5fa98b771e745ffcb0b8c886771038da9"
SHA256 = "dc16f1194989a38cb056ba293d428cd9fe5cbdcddf5f419112c5763c2233fd09"
DEFAULT_OUTPUT = Path.home() / ".cache/twi-pipecat/sunflower-v2-mlx"


def read_tensors(path: Path) -> dict[str, np.ndarray]:
    tensors = {}
    dtypes = {0: "<f4", 1: "i1", 2: "<i2", 3: "<i4", 4: "<f2"}
    with path.open("rb") as stream:

        def number(fmt):
            return struct.unpack("<" + fmt, stream.read(struct.calcsize(fmt)))[0]

        def string():
            return stream.read(number("H")).rstrip(b"\0").decode()

        if (number("I"), string(), number("I")) != (6, "WhisperSpec", 3):
            raise ValueError("Unsupported checkpoint format")
        for _ in range(number("I")):
            name = string()
            shape = tuple(number("I") for _ in range(number("B")))
            dtype = np.dtype(dtypes[number("B")])
            size = number("I")
            if size != int(np.prod(shape)) * dtype.itemsize:
                raise ValueError(f"Invalid tensor size: {name}")
            tensors[name] = np.memmap(
                path, mode="r", dtype=dtype, offset=stream.tell(), shape=shape
            )
            stream.seek(size, 1)
        for _ in range(number("I")):
            alias, source = string(), string()
            tensors[alias] = tensors[source]
        if stream.read(1):
            raise ValueError("Unexpected trailing checkpoint data")
    return tensors


def convert(output: Path):
    source = Path(
        hf_hub_download(REPO, "model.bin", revision=REVISION, local_files_only=True)
    ).parent
    with (source / "model.bin").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
            raise ValueError("Source checkpoint checksum mismatch")
    tensors = read_tensors(source / "model.bin")
    weights = {}

    def put(destination, value):
        if value.dtype != np.float16:
            raise ValueError(f"Expected FP16 weights: {destination}")
        weights[destination] = mx.array(np.array(value))

    def copy(destination, source):
        put(destination, tensors[source])

    def norm(destination, source):
        copy(destination + ".weight", source + "/gamma")
        copy(destination + ".bias", source + "/beta")

    for side in ("encoder", "decoder"):
        for index in range(32):
            src, dst = f"{side}/layer_{index}", f"{side}.blocks.{index}"
            norm(dst + ".attn_ln", src + "/self_attention/layer_norm")
            norm(dst + ".mlp_ln", src + "/ffn/layer_norm")
            for parameter in ("weight", "bias"):
                q, k, v = np.split(tensors[f"{src}/self_attention/linear_0/{parameter}"], 3)
                for name, value in (("query", q), ("key", k), ("value", v)):
                    if name == "key" and parameter == "bias":
                        if np.any(value):
                            raise ValueError("MLX does not support nonzero attention key bias")
                    else:
                        put(f"{dst}.attn.{name}.{parameter}", value)
                copy(f"{dst}.attn.out.{parameter}", f"{src}/self_attention/linear_1/{parameter}")
                for n in range(2):
                    copy(f"{dst}.mlp{n + 1}.{parameter}", f"{src}/ffn/linear_{n}/{parameter}")
                if side == "decoder":
                    copy(
                        f"{dst}.cross_attn.query.{parameter}",
                        f"{src}/attention/linear_0/{parameter}",
                    )
                    k, v = np.split(tensors[f"{src}/attention/linear_1/{parameter}"], 2)
                    put(f"{dst}.cross_attn.value.{parameter}", v)
                    if parameter == "weight":
                        put(f"{dst}.cross_attn.key.{parameter}", k)
                    elif np.any(k):
                        raise ValueError("Nonzero cross-attention key bias")
                    copy(
                        f"{dst}.cross_attn.out.{parameter}", f"{src}/attention/linear_2/{parameter}"
                    )
            if side == "decoder":
                norm(dst + ".cross_attn_ln", src + "/attention/layer_norm")
    for n in (1, 2):
        put(f"encoder.conv{n}.weight", tensors[f"encoder/conv{n}/weight"].transpose(0, 2, 1))
        copy(f"encoder.conv{n}.bias", f"encoder/conv{n}/bias")
    norm("encoder.ln_post", "encoder/layer_norm")
    norm("decoder.ln", "decoder/layer_norm")
    copy("decoder.token_embedding.weight", "decoder/embeddings/weight")
    copy("decoder.positional_embedding", "decoder/position_encodings/encodings")
    if not np.array_equal(
        tensors["decoder/projection/weight"], tensors["decoder/embeddings/weight"]
    ):
        raise ValueError("MLX requires tied decoder output embeddings")

    dims = dict(
        n_mels=128,
        n_audio_ctx=1500,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        n_vocab=51866,
        n_text_ctx=448,
        n_text_state=1280,
        n_text_head=20,
        n_text_layer=32,
    )
    model = Whisper(ModelDimensions(**dims), mx.float16)
    config = json.loads((source / "config.json").read_text())
    weights["alignment_heads"] = mx.array(config["alignment_heads"])
    model.set_alignment_heads(np.array(config["alignment_heads"]))
    expected = dict(tree_flatten(model.parameters()))
    if set(weights) != set(expected):
        raise ValueError(f"Weight names differ: {set(weights) ^ set(expected)}")
    for name, value in weights.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"Weight shape mismatch: {name}")
    # Unlike base Whisper, Sunflower fine-tuned its encoder positions. MLX's
    # loader uses Module.update(), which accepts this existing private array;
    # persist it explicitly rather than silently regenerating sinusoids.
    copy("encoder._positional_embedding", "encoder/position_encodings/encodings")
    if weights["encoder._positional_embedding"].shape != model.encoder._positional_embedding.shape:
        raise ValueError("Encoder position shape mismatch")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "weights.pending.safetensors"
    mx.save_safetensors(str(temporary), weights)
    temporary.replace(output / "weights.safetensors")
    (output / "config.json").write_text(json.dumps(dims, indent=2))
    (output / "conversion.json").write_text(
        json.dumps(
            dict(
                source=REPO,
                revision=REVISION,
                sha256=SHA256,
                precision="float16",
                language_tokens={"aka": "jw", "eng": "en"},
                tensors=len(weights),
            ),
            indent=2,
        )
    )
    print(f"Validated and converted {len(weights)} tensors to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    convert(parser.parse_args().output)
