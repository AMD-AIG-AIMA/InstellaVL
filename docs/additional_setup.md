## How to support `InstellaVL` evaluation in `lmms-eval`?

1. Clone [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repo, just *outside* (i.e., adjacent to InstellaVL repository). And then go to the specified commit.
    ```bash
    # Assuming you are just outside of InstellaVL repo
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    cd lmms-eval
    git checkout 8b68660431a50024f6775ae468c70d074e224c9d
    ```
2. Say, you have cloned the `lmms-eval` repo just outside `InstellaVL`, now do the following.
    1. ```bash
       # Assuming you are just inside InstellaVL repo
       cp assets/patches/instellavl.py ../lmms-eval/lmms_eval/models/.
       ```
    2. Further add this line `"instellavl": "InstellaVL",` to `AVAILABLE_MODELS` dictionary of `lmms-eval/lmms_eval/models/__init__.py` file.
That's it :relaxed:. Now follow the [**evaluation instruction**](../README.md#straight_ruler-evaluation) for evaluating InstellaVL on pre-mentioned benchmarks.

> [!NOTE]
> Please bear with this patch until our PR gets merge into the original repository of `lmms-eval`.
