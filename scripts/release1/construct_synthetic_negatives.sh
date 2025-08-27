#########################################################################################################
##################### Synthetic Negatives -- Preprocess
#########################################################################################################

### Note: Since the underlying "open-thoughts/OpenThoughts2-1M" could change, do not rely on this script
### for the mapping to the instance id's. Instead, use the text of the OpenVerification1 dataset, which
### has already preprocessed the data to remove the thinking traces. We provide this only for reference for
### what the structure of the data looks like for the next script.

conda activate preprocess_env1

cd code/data_processing/release1

export HF_HOME=  # set

OUTPUT_DIR= ... [UPDATE] ... data/original_data

python -u openthoughts_synthetic_negatives__step1_prepare_data.py \
--seed_value 0 \
--output_file "${OUTPUT_DIR}/OpenThoughts2-1M.jsonl" \
--output_file_streamlined "${OUTPUT_DIR}/OpenThoughts2-1M_shuffled_streamlined.jsonl"

#Total unexpected formatting so skipped: 85


#########################################################################################################
##################### Synthetic Negatives -- Preprocess -- shard 3 (marked with 'v3')
## This would correspond to the meta="openthoughts.shard3" instances in OpenVerification1.
#########################################################################################################

### Note: To make this easier to follow, we show how to construct the negatives as if you were calling the standard streaming APIs. In practice to save costs and time, it is recommended that you upload shards of the data for batch prediction for dataset construction. Consult the Azure/OpenAI/etc. documentation for details.

source llm_apis.sh  # applicable API keys

conda activate preprocess_env1

cd code/data_processing/release1

INPUT_DIR=... [UPDATE] ... data/original_data
OUTPUT_DIR=... [UPDATE] ... data/openthoughts/v3
mkdir ${OUTPUT_DIR}

python -u openthoughts_synthetic_negatives.py \
--input_file "${INPUT_DIR}/OpenThoughts2-1M_shuffled_streamlined.jsonl" \
--start_index 20000 \
--total_lines 3000 \
--shards 20 \
--output_dir ${OUTPUT_DIR}

#########################################################################################################
##################### Construct combined data splits
## This would correspond to the meta="openthoughts.shard3" instances in OpenVerification1.
#########################################################################################################

OUTPUT_PROCESSED_COMBINED_DIR="... [UPDATE] ... /data/openthoughts/v3_combined"
mkdir ${OUTPUT_PROCESSED_COMBINED_DIR}

# Next, we combine into one file:
OUTPUT_PROCESSED_DIR="... [UPDATE] ... /data/openthoughts/v3"
cat ${OUTPUT_PROCESSED_DIR}/openthoughts_synthetic_neg*.jsonl > ${OUTPUT_PROCESSED_COMBINED_DIR}/openthoughts_synthetic_shard3.jsonl


#########################################################################################################
##################### Convert multiple choice to binary verification
## For the multiple choice datasets, we simply randomly select an incorrect letter.
#########################################################################################################

conda activate preprocess_env1

cd code/data_processing/release1

export HF_HOME=  # set

OUTPUT_DIR=... [UPDATE] ... /data/multiple_choice/step1

python -u binary_multiple_choice__step1_prepare_data.py \
--seed_value 0 \
--output_dir "${OUTPUT_DIR}"

#########################################################################################################
##################### Convert FEVER to binary
## As with the multiple choice datasets, construction of the examples for the FEVER data does not
## involve an LLM. In practice, the FEVER examples are rather simplistic and formulaic, and are of
## interest primarily from the perspective of an earlier era of NLP research. In practice,
## we did not use them in version 1.1.1 or later of the SDM estimator in the Reexpress MCP Server.
#########################################################################################################

conda activate preprocess_env1

cd code/data_processing/release1

export HF_HOME=  # set

OUTPUT_DIR=... [UPDATE] ... /data/fever/step1
mkdir -p ${OUTPUT_DIR}

python -u binary_fever__step1_prepare_data.py \
--seed_value 0 \
--output_dir "${OUTPUT_DIR}"


#########################################################################################################
##################### NEXT STEPS after constructing synthetic negatives: Generate verification output
#########################################################################################################

# Use the LLMs, prompts, and parameters of [`mcp_utils_llm_api.py`](https://github.com/ReexpressAI/reexpress_mcp_server/blob/main/code/reexpress/mcp_utils_llm_api.py) of the Reexpress MCP Server. For dataset construction, we recommend using the applicable Batch Prediction APIs, which are typically signficantly less expensive than directly using the streaming APIs.

#########################################################################################################
##################### NEXT STEPS after generating the verification output: Generate hidden states of an agreement model
#########################################################################################################

# To generate the hidden states of an agreement model, see, for example, get_model_explanations_formatted_as_binary_agreement_prompt() and get_agreement_model_embedding() as used in llm_api_controller() of [`mcp_utils_llm_api.py`](https://github.com/ReexpressAI/reexpress_mcp_server/blob/main/code/reexpress/mcp_utils_llm_api.py) of the Reexpress MCP Server.
