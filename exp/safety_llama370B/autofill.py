import json
import random


# choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
choices = ["safe", "unsafe"]
with open("output_binary.json") as fin:
    data = json.load(fin)

for datapiece in data:
    if datapiece["llama370B"] == "":
        datapiece["llama370B"] = random.choice(choices)

with open("output_binary.json", "w") as fout:
    json.dump(data, fout, indent=4, ensure_ascii=False)
