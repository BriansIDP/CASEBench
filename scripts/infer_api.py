import json
import random
import argparse
import os, sys
import re
import math
import time
from copy import deepcopy
import regex

from openai import OpenAI
from tqdm import tqdm

from template import judge, judgescore

# Cost table: <model_name>: [completion_price_per_1k_token, prompt_price_per_1k_token]
costdict = {
    "gpt-3.5-turbo-0125": [0.0015, 0.0005],
    "gpt-3.5-turbo-1106": [0.0015, 0.0005],
    "gpt-4-1106-preview": [0.03, 0.01],
    "gpt-4-0125-preview": [0.03, 0.01],
    "gpt-4o": [0.015, 0.005],
    "gpt-4o-mini-2024-07-18": [0.00015, 0.00005],
    "o1-preview-2024-09-12": [0.060, 0.015],
    "o1-mini-2024-09-12": [0.012, 0.003]
}


def request(prompt, total_cost, client, mode="binary"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=1024,
        temperature=1.0,
    )
    response = json.loads(response.model_dump_json())
    turn_cost = costdict[args.model][0] * response["usage"]["completion_tokens"] / 1000 + (
        costdict[args.model][1] * response["usage"]["prompt_tokens"]) / 1000
    total_cost += turn_cost
    content = response["choices"][0]["message"]["content"]
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
    return content, total_cost


def main(args):
    api_key = os.environ.get("OPENAI_API_KEY", None)
    client = OpenAI(api_key=api_key)

    with open(args.infile) as fin:
        data = json.load(fin)

    results = []
    total_cost = 0
    for datapiece in tqdm(data):
        if args.model not in datapiece or datapiece[args.model] == "":
            query = datapiece["query"]

            # Remove a particular parameter
            if args.remove != "":
                if args.remove in datapiece["context"]:
                    datapiece["context"].pop(args.remove)
                elif args.remove in datapiece["context"]["transmission_principle"]:
                    datapiece["context"]["transmission_principle"].pop(args.remove)
                else:
                    raise Exception("Not a valid parameter: {}".format(args.remove))

            context = json.dumps(datapiece["context"])
            if args.mode == "binary":
                prompt = judge.format(query=query, context=context)
            else:
                prompt = judgescore.format(query=query, context=context)

            if args.mode == "binary":
                judgement, total_cost = request(prompt, total_cost, client, mode=args.mode)
            else:
                judgements = []
                for i in range(3):
                    judgement, total_cost = request(prompt, total_cost, client, mode=args.mode)
                    if judgement != "":
                        judgements.append(judgement)
                if judgements != []:
                    judgement = sum(judgements) / len(judgements)
                else:
                    judgement = ""
            datapiece[args.model] = judgement
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
