
import os
from os import PathLike
import logging

from models import DecoderBase
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

from data import load_problems

logger = logging.getLogger("my_logger")


def construct_contract_prompt(prompt: str, contract_type: str, contract: str) -> str:
    if contract_type == "none":
        return prompt
    elif contract_type == "docstring":
        # embed within the docstring
        sep = ""
        if '"""' in prompt:
            sep = '"""'
        elif "'''" in prompt:
            sep = "'''"
        assert sep != ""
        l = prompt.split(sep)
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        l[1] = (
            l[1] + contract + "\n" + " " * (len(contract) - len(contract.lstrip()) - 1)
        )
        return sep.join(l)
    elif contract_type == "code":
        # at the beginning of the function
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        return prompt + contract


def construct_prompt(args, problem: dict) -> str:
    prompt = """Please solve the following programming problem entitled "{title}" in {lang}, the problem is described below:

{description}


{examples}


{constraints}


Please use the following code template:

```{lang_slug}
{code_template}
```


Note that you only need to complete the code template above, and do not need to provide the code on how to use it.
""".format(
        title=problem["title"],
        lang=args.lang,
        description=problem["description"],
        examples="**Examples:**\n\n" + "\n".join([f"Example {i}:\n{example}\n" for i, example in enumerate(problem["examples"])]),
        constraints=problem["constraints"],
        lang_slug=args.lang_slug,
        code_template=problem["templates"][args.lang_slug]
    )
    if args.model == "gpt-4-1106-preview":
        prompt += '\n\nPlease response by generating JSON like {"code": ""}'
    return prompt


def code_generate(args, model: DecoderBase):
    with Progress(
        TextColumn(
            f"{args.model} •" + "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as p:

        problems = load_problems(args)

        for problem in p.track(problems):

            p_name = problem["id"]

            problem_dir = os.path.join(args.output_root, "model_outputs", p_name)
            os.makedirs(problem_dir)

            logger.info(f"Generating code for {p_name}")

            for idx in range(args.n_samples):

                prompt = construct_prompt(args, problem)

                outputs = model.codegen(
                    prompt,
                    do_sample=not args.greedy,
                    num_samples=args.n_samples,
                )

                assert outputs, "No outputs from model!"
                for impl in outputs:
                    try:
                        with open(os.path.join(problem_dir, f"{idx}.{args.lang_ext}"), mode="w", encoding="utf-8") as f:
                            f.write(impl)
                    except UnicodeEncodeError:
                        continue
