import json
import os
from abc import ABC, abstractmethod
from typing import List
from warnings import warn
import logging

import openai
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from vllm import LLM, SamplingParams

from utils import make_auto_request

EOS = ["<|endoftext|>", "<|endofmask|>", "</s>"]
HF_HOME = ""
logger = logging.getLogger("my_logger")


# Adopted from https://github.com/huggingface/transformers/pull/14897
class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length:]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                    finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length: -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


class DecoderBase(ABC):
    def __init__(
            self,
            name: str,
            batch_size: int = 1,
            temperature: float = 0.8,
            max_new_tokens: int = 2048,
            conversational: bool = False,
            max_conversational_new_tokens: int = 2048,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = (
            max_conversational_new_tokens if conversational else max_new_tokens
        )
        self.conversational = conversational

    @abstractmethod
    def codegen(
            self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class VLlmDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

        # kwargs = {"tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", "1"))}
        kwargs = {
            "tensor_parallel_size": int(torch.cuda.device_count()),
            "max_model_len": 37040  # V100 32G
        }
        if "CodeLlama" in name:
            # kwargs["dtype"] = "bfloat16"
            kwargs["dtype"] = "half"
        elif "CodeBooga" in name:
            kwargs["dtype"] = "float16"
        elif "WizardCoder" in name:
            kwargs["dtype"] = "float16"
        elif "deepseek" in name:
            # kwargs["dtype"] = "bfloat16"
            kwargs["dtype"] = "half"
            kwargs["trust_remote_code"] = True
        elif "mixtral" in name.lower():
            kwargs["dtype"] = "bfloat16"
        elif "solar" in name:
            kwargs["dtype"] = "float16"
        elif "mistral" in name.lower():
            kwargs["dtype"] = "bfloat16"
        elif "phi" in name.lower():
            kwargs["dtype"] = "float16"
            kwargs["max_model_len"] = 2048
            kwargs["trust_remote_code"] = True

        self.llm = LLM(model=name, **kwargs)

    def codegen(
            self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        if "deepseek-coder" in self.name and "instruct" in self.name:
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop_token_ids=[32021],
            )
        else:
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            )

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            sampling_params,
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs


class WizardCoderDecoder(VLlmDecoder):
    def codegen(
            self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
{prompt}

### Response:"""

        return VLlmDecoder.codegen(self, prompt, do_sample, num_samples)


class DeepSeekInstructVLlmDecoder(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]

        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def codegen(
            self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        messages = [
            {
                'role': 'user',
                "content": prompt
            }
        ]
        input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        return VLlmDecoder.codegen(self, input, do_sample, num_samples)


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = openai.OpenAI(
            api_key=open("openai.key", mode="r", encoding="utf-8").read().strip(),
            base_url="https://api.chatanywhere.com.cn/v1"
        )

    def codegen(
            self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        assert batch_size <= 20, "Use larger batch size could blow up the memory!"

        # # construct prompt
        fmt = "json_object" if self.name == "gpt-4-1106-preview" else "text"

        message = prompt

        ret = make_auto_request(
            self.client,
            message=message,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
            response_format={"type": fmt},
        )

        outputs = []
        for item in ret.choices:
            content = item.message.content
            # if json serializable
            if fmt == "json_object":
                try:
                    json_data = json.loads(content)
                    if json_data.get("code", None) is not None:
                        outputs.append(json_data["code"])
                        continue

                    print(f"'code' field not found in: {json_data}")
                except Exception as e:
                    print(e)
            outputs.append(content)

        return outputs


def make_model(name: str, batch_size: int = 1, temperature: float = 0.8):

    if name.startswith("gpt-3.5-") or name.startswith("gpt-4-"):
        return OpenAIChatDecoder(
            batch_size=batch_size,
            name=name,
            temperature=temperature,
            conversational=True,
        )
    elif name.startswith("code-llama-"):
        assert name.endswith("b")
        nb = name.split("-")[-1]
        return VLlmDecoder(
            batch_size=batch_size,
            name=HF_HOME + f"codellama/CodeLlama-{nb}-Python-hf",
            temperature=temperature,
        )
    elif name.startswith("deepseek-coder"):
        import re

        # format deepseek-coder-{nb}b*
        pattern = re.compile(r"deepseek-coder-(\d+\.?\d*)b(.*)")
        matches = pattern.findall(name)[0]
        nb = float(matches[0])
        if nb.is_integer():
            nb = int(nb)

        if "instruct" in name:
            return DeepSeekInstructVLlmDecoder(
                batch_size=batch_size,
                name=HF_HOME + f"deepseek-ai/deepseek-coder-{nb}b-instruct",
                temperature=temperature,
                conversational=True,
            )
    elif name == "wizardcoder-34b":
        return WizardCoderDecoder(
            batch_size=batch_size,
            name=HF_HOME + "WizardLM/WizardCoder-Python-34B-V1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "wizardcoder-15b":
        return WizardCoderDecoder(
            batch_size=batch_size,
            name=HF_HOME + "WizardLM/WizardCoder-15B-V1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "wizardcoder-7b":
        return WizardCoderDecoder(
            batch_size=batch_size,
            name=HF_HOME + "WizardLM/WizardCoder-Python-7B-V1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "wizardcoder-13b":
        return WizardCoderDecoder(
            batch_size=batch_size,
            name=HF_HOME + "WizardLM/WizardCoder-Python-13B-V1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "phi-2":
        return VLlmDecoder(
            batch_size=batch_size,
            name=HF_HOME + "microsoft/phi-2",
            temperature=temperature,
        )

    raise ValueError(f"Invalid model name: {name}")
