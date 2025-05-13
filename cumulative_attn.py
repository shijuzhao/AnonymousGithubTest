import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from benchmarks_infoblend.test_type.prob_attn.configs import Configs
from benchmarks_infoblend.utils import set_seed
from benchmarks_infoblend.data.utils import str2class

def get_args() -> argparse.Namespace:
    """ Get arguments from the command line.
    Please refer to the configs.py file for detailed information.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='example', help="The dataset name")
    parser.add_argument("--model", type=str, default='benchmarks_infoblend/models/llava-v1.6-vicuna-7b-hf', help="The model name from huggingface")
    args = parser.parse_args()

    return args

def run(configs, dataset, llm, tokenizer):
    # Start testing
    for system_prompts, mod_prompts, free_form_prompt in dataset:
        images = [Image.open(url) for url in mod_prompts]
        image_spec_token = '<image>'
        # InternVL have different name of image token.
        if 'Intern' in configs.model:
            image_spec_token = '<IMG_CONTEXT>'
        image_id = tokenizer.convert_tokens_to_ids(image_spec_token)
        attn_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.attn_metadata
        attn_metadata['collect'] = True
        sampling_params = SamplingParams(
            temperature=0, 
            max_tokens=2,
        )
        old_kvs = []
        output = llm.generate(
            {
                "prompt": system_prompts+free_form_prompt,
                "multi_modal_data": {
                    "image": images
                },
            },
            sampling_params=sampling_params, use_tqdm=False)

        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        for j in range(len(llm_layers)):
            temp_k = llm_layers[j].self_attn.kv[0].clone()
            temp_v = llm_layers[j].self_attn.kv[1].clone()     
            old_kvs.append([temp_k, temp_v])

        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = old_kvs
        attn_metadata['prob_attn'] = True
        output = llm.generate(
            {
                "prompt": system_prompts + free_form_prompt,
                "multi_modal_data": {
                    "image": images
                },
            },
            sampling_params=sampling_params, use_tqdm=False)
        
        image_token_num = int(output[0].prompt_token_ids.count(image_id) / len(images))
        result_folder = os.path.join(configs.cwd, 'cumulative_attn')
        os.makedirs(result_folder, exist_ok=True)
        start_image_token = output[0].prompt_token_ids.index(image_id)
        fontsize = 20
        plt.figure("line plot", figsize=(4.5, 6))
        x_coords = torch.arange(1, image_token_num + 1)
        for j in range(len(llm_layers)):
            attn_matrix = attn_metadata['attn_matrix'][j][0].squeeze()
            # reshape for GQA (grouped query attention)
            attn_matrix = attn_matrix.reshape(-1, attn_matrix.shape[-1])
            culmulative_attn = torch.zeros(image_token_num)
            for i in range(attn_matrix.shape[0]):
                image_score = attn_matrix[i].cpu().float()[start_image_token: start_image_token+image_token_num]
                attn_score = F.softmax(image_score, dim=0)
                for idx in range(1, len(attn_score)):
                    attn_score[idx] += attn_score[idx - 1]
                culmulative_attn += attn_score
            plt.plot(x_coords, culmulative_attn / attn_matrix.shape[0])
            plt.xlabel('n', fontsize=fontsize)
            plt.ylabel('Sum of attention scores of the first n tokens', fontsize=fontsize-8)
            plt.savefig(configs.cwd + f'/cumulative_attn/layer_{j}.png', bbox_inches='tight')
            plt.clf()
        
if __name__ == "__main__":

    args = get_args()
    configs = Configs(
        dataset=args.dataset,
        model=args.model
    )

    tokenizer = AutoTokenizer.from_pretrained(configs.model, add_bos_token=True, trust_remote_code=True)
    dataset = str2class[configs.dataset](tokenizer=tokenizer, path=configs.result_folder)

    set_seed(configs.seed)
    llm = LLM(
        model=configs.model,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        max_model_len=4096,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2}
    )
    llm.set_tokenizer(tokenizer)
    run(configs, dataset, llm, tokenizer)