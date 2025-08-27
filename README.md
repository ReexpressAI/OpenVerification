# OpenVerification dataset

This repo contains some ancillary preprocessing scripts and parameters for the OpenVerification series of datasets. The first release, [OpenVerification1](https://huggingface.co/datasets/ReexpressAI/OpenVerification1), is available via HuggingFace Datasets.

Refer to the [Dataset Card for OpenVerification1](https://huggingface.co/datasets/ReexpressAI/OpenVerification1/blob/main/README.md) for an overview and the key details of the dataset. To stay in-distribution[^1] to the models, prompts, and parameters of the classifications in the dataset, use the LLMs, prompts, and parameters of [`mcp_utils_llm_api.py`](https://github.com/ReexpressAI/reexpress_mcp_server/blob/main/code/reexpress/mcp_utils_llm_api.py) of the Reexpress MCP Server.

In contrast, this repo refers to the generation of synthetic negatives provided in the "Additional Reference Fields" in the dataset. For typical use-cases of the dataset, this additional background information is not needed.

See the [overview script](scripts/release1/construct_synthetic_negatives.sh) to step through the process of constructing the synthetic negatives.

More generally, as a high-level overview, the dataset was created by:

1. Constructing synthetic negatives (detailed in this repo)
2. Generating classifications, uncertainty estimates, and explanations over the label=0 and label=1 instances. See [`mcp_utils_llm_api.py`](https://github.com/ReexpressAI/reexpress_mcp_server/blob/main/code/reexpress/mcp_utils_llm_api.py) of the Reexpress MCP Server.
3. [Separately for the Reexpress MCP Server, we use an additional model to compose the output from the LLMs from Step 2. The classifications from Step 2 and the hidden states from this additional model are then the input to an SDM estimator to determine the predictive uncertainty.]

> [!TIP]
> We primarily provide this for researchers interested in generating additional synthetic negatives. Generally speaking, we recommend *not* using these prompts, this model (`gpt-4.1-2025-04-14`), nor these parameters for constructing additional negative examples in order to create additional variance in the distribution of negatives. In particular, it is recommended to use an altogether different model than `gpt-4.1-2025-04-14`.

[^1]: Alternatively, if you want to change the models, prompts, and/or generation parameters of the classifications (over label=0 and label=1 instances) already in the dataset, you should create four new columns for that output: modelNAME; modelNAME_verification_classification; modelNAME_confidence_in_classification; modelNAME_short_explanation_for_classification_confidence (or comparable information if you are also changing the structure of the model output itself). 
