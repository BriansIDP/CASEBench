import json
import random
import argparse
import os, sys
import re
import math
import time
from copy import deepcopy
import regex

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from template import judge, judgescore, judgelogits


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

safe_dict = {
    "llama370B": [19193, 6220, 26747, 23088],
    "qwen272B": [18675, 6092, 25663, 22291],
    "mixtral8x7B": [5023, 24092],
    "dolphin": [19193, 6220, 26747, 23088],
}

unsafe_dict = {
    "llama370B": [39257, 20451, 79870, 74167],
    "qwen272B": [38157, 19860, 78770, 73067],
    "mixtral8x7B": [15426, 935],
    "dolphin": [39257, 20451, 79870, 74167],
}

def request(prompt, tokenizer, model, mode="binary", modelname="llama3"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if "llama3" in modelname:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        content = tokenizer.decode(response, skip_special_tokens=True)
    elif "mixtral" in modelname or "dolphin" in modelname:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
        )
        response = outputs[0][input_ids.shape[-1]:]
        content = tokenizer.decode(response, skip_special_tokens=True)
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        content = response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    try:
        pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
        content = pattern.findall(content)
        # assert len(content) == 1
        content = json.loads(content[0])
        if mode == "score":
            content = float(content["judgement"])
        else:
            if "safe" in content["judgement"] and "unsafe" not in content["judgement"]:
                content = "safe"
            elif "unsafe" in content["judgement"]:
                content = "unsafe"
            else:
                content = ""
    except:
        content = ""
        pass
    return content


def request_logits(prompt, tokenizer, model, modelname="llama3"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if "llama3" in modelname:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = model.generate(
            input_ids,
            max_new_tokens=8,
            eos_token_id=terminators,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
        prob_dist = torch.softmax(outputs['logits'][0], dim=-1)[0]
        safe_prob = prob_dist[safe_dict[modelname]].sum()
        unsafe_prob = prob_dist[unsafe_dict[modelname]].sum()
        prob = torch.stack([safe_prob, unsafe_prob])
        prob = prob / prob.sum()
        content = prob[0].tolist()
    elif "mixtral" in modelname or "dolphin" in modelname:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        outputs = model.generate(
            input_ids,
            max_new_tokens=8,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
        prob_dist = torch.softmax(outputs['logits'][0], dim=-1)[0]
        safe_prob = prob_dist[safe_dict[modelname]].sum()
        unsafe_prob = prob_dist[unsafe_dict[modelname]].sum()
        prob = torch.stack([safe_prob, unsafe_prob])
        prob = prob / prob.sum()
        content = prob[0].tolist()
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=8,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
        prob_dist = torch.softmax(generated_ids['logits'][0], dim=-1)[0]
        safe_prob = prob_dist[safe_dict[modelname]].sum()
        unsafe_prob = prob_dist[unsafe_dict[modelname]].sum()
        prob = torch.stack([safe_prob, unsafe_prob])
        prob = prob / prob.sum()
        content = prob[0].tolist()
    return content


def main(args):
    model_mapping = {
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama370B": "meta-llama/Meta-Llama-3-70B-Instruct",
        "qwen272B": "Qwen/Qwen2-72B-Instruct",
        "mixtral8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "dolphin": "cognitivecomputations/dolphin-2.9-llama3-70b",
    }

    ## Initialise model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_mapping[args.model],
        torch_dtype=torch.bfloat16,
        cache_dir="",
        device_map="auto",
    )
    # model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_mapping[args.model],
        cache_dir="",
    )

    with open(args.infile) as fin:
        data = json.load(fin)

    results = []
    total_cost = 0
    for datapiece in tqdm(data):
        if args.model not in datapiece or datapiece[args.model] == "":
            query = datapiece["query"]

            # Remove a particular parameter
            if args.remove != "":
                remove_strs = args.remove.split(",")
                for remove_str in remove_strs:
                    if remove_str in datapiece["context"]:
                        datapiece["context"].pop(remove_str)
                    elif remove_str in datapiece["context"]["transmission_principle"]:
                        datapiece["context"]["transmission_principle"].pop(remove_str)
                    else:
                        raise Exception("Not a valid parameter: {}".format(args.remove))

            context = json.dumps(datapiece["context"])
            if args.mode == "binary":
                prompt = judge.format(query=query, context=context)
            elif args.mode == "logits":
                prompt = judgelogits.format(query=query, context=context)
            else:
                prompt = judgescore.format(query=query, context=context)

            if args.mode == "binary":
                judgement = request(prompt, tokenizer, model, mode=args.mode, modelname=args.model)
            elif args.mode == "logits":
                judgement = request_logits(prompt, tokenizer, model, modelname=args.model)
            else:
                judgements = []
                for i in range(1):
                    judgement = request(prompt, tokenizer, model, mode=args.mode, modelname=args.model)
                    if judgement != "":
                        judgements.append(judgement)
                if judgements != []:
                    judgement = sum(judgements) / len(judgements)
                else:
                    judgement = ""
            datapiece[args.model] = judgement
            with open(args.outfile, "w") as fout:
                json.dump(data, fout, indent=4, ensure_ascii=False)
        results.append(datapiece)

    # gen_context["total_cost"] = total_cost
    print(total_cost)
    with open(args.outfile, "w") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Generation")
    parser.add_argument(
        "--infile",
        type=str,
        default="./data/categorised_topics.json",
        help="Path to the topic file",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="./data/train_10k.jsonl",
        help="Path to the conversation file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="Generation model type",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="binary",
        help="Whether to use binary or score",
    )
    parser.add_argument(
        "--remove",
        type=str,
        default="",
        help="which part of the context to be removed",
    )
    args = parser.parse_args()
    main(args)
