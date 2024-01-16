import json
import logging


logger = logging.getLogger("my_logger")


def load_problems(args) -> list[dict]:
    assert args.input_file.endswith(".jsonl")
    problems = []
    with open(args.input_file, mode="r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))
    return problems
