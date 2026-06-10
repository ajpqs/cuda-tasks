# RmsNormGated

Implement a component for the [Qwen 3 Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) LLM architecture: [RmsNormGated](https://github.com/huggingface/transformers/blob/96c41be562d43785d07744924c2f1e96bc7d6884/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L69-L84) for fp16 data type.

`HeadSize` is guaranteed to be a power of two and at least 8.
