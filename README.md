# EfficiencyEval

This repository contains two parts, evalplus and LeetCodeEval.

## evalplus

This is a modified version of [evalplus](https://github.com/evalplus/evalplus). The usage is the same, and the result file will have runtime in it.

## LeetCodeEval

This directory contains LeetCodeEval dataset and the source code for generating response from LLMs.

### Dataset

LeetCodeEval dataset is in the `leetcode` direcotory, each line is LeetCode problem in json format.

### Usage

Run `main.py` to start generation. All arguments are located in `main.py`, specific whatever you need.

Some example scripts are as following.

```sh
# run gpt-3.5-turbo-1106 on medium set with temperture=0.8 and n=3
--model gpt-3.5-turbo-1106 \
--bs 3 \
--temperature 0.8 \
--input_file leetcode/medium.jsonl \
--n_samples 3

# run deepseek-coder-33b-instruct on easy set with temperture=0.8 and n=3
python main.py \
--model deepseek-coder-33b-instruct \
--bs 3 \
--temperature 0.8 \
--input_file leetcode/easy.jsonl \
--n_samples 3
```
