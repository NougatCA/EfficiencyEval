import os
import time
import logging
import argparse
from rich.logging import RichHandler

from models import make_model
from generate import code_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--greedy", action="store_true")

    parser.add_argument("--lang", default="C++", type=str)
    parser.add_argument("--lang_slug", default="cpp", type=str)
    parser.add_argument("--lang_ext", default="cpp", type=str)

    args = parser.parse_args()

    input_name = os.path.splitext(args.input_file)[0].replace("/", "_")
    args.run_name = f"{args.model}_{input_name}_{args.lang_slug}_temp-{args.temperature}"
    args.output_root = os.path.join(
        "outputs", f"{args.run_name}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    )
    os.makedirs(args.output_root)

    # logging, log to both console and file, log debug-level to file
    logger = logging.getLogger("my_logger")
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        log_time_format="[%X]",
        level=logging.INFO
    ))
    # logging file
    file = logging.FileHandler(os.path.join(args.output_root, "logging.log"))
    file.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s | %(filename)s: %(lineno)d] - %(levelname)s: %(message)s"
    )
    file.setFormatter(formatter)
    logger.addHandler(file)

    if args.greedy and (args.temperature != 0 or args.bs != 1 or args.n_samples != 1):
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        logger.warning("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    model = make_model(
        name=args.model, batch_size=args.bs, temperature=args.temperature
    )

    with open(os.path.join(args.output_root, "args.txt"), "w") as f:
        f.write(str(args))

    code_generate(args, model=model)


if __name__ == "__main__":
    main()
