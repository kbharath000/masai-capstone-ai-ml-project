# Vendored embedding model

`all-MiniLM-L6-v2/` contains the files `sentence-transformers.SentenceTransformer` needs to
load the model with **no network call**: `model.safetensors`, tokenizer files, and the
sentence-transformers config/pooling json files. Non-essential formats from the original
repo (`pytorch_model.bin`, `tf_model.h5`, `rust_model.ot`, `onnx/`, `openvino/`, docs) were
left out to keep this directory small (~87 MB instead of ~1 GB).

This is vendored rather than downloaded at runtime because `huggingface.co` (and any
`*.hf.co` CDN host, including mirrors that redirect to it) is blocked by Pfizer's corporate
proxy (category `pfe_huggingface`) on the network this project was built on. The weights
were fetched once via ModelScope's mirror of `sentence-transformers/all-MiniLM-L6-v2`
(`modelscope.cn`, not blocked) and committed here so both local runs and the Docker image
work offline.

`support_assistant.py` points `EMBEDDING_MODEL_NAME` at this directory instead of the model
name string, so `SentenceTransformer(EMBEDDING_MODEL_NAME)` loads it as a local path.
