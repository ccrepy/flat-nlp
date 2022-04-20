# Flat NLP

*"Feature Language As Trajectories"* is a paradigm of NLP where sentences are
considered as trajectories (as opposed as points traditionally).

## Manifest

Modifications compared to the original codebase:

* **`flat_nlp/public/queryset_util.py`** has been modified:

  * `build_tokenizer_fn(locale: str) -> _TokenizerFn` could not be opensourced.

* **`flat_nlp/encoding/encoder_factory.py`** has been modified:

  * `_load_pretrained_embedding_encoder(config: flat_pb2.EncoderConfig.PretrainedEmbeddingEncoderConfig) -> encoder_util.Encoder` could not be opensourced.

* **`flat_nlp/encoding/lib/testdata/text8.5.vec`** is not uploaded.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
