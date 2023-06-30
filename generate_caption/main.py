import argparse
import torch

from fastchat.model import load_model
from accelerate import PartialState
from dataset import CaptionDataset
from torch.utils.data import DataLoader


def main(args):
    # load model
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision,)

    # load data
    caption_dataset = CaptionDataset(caption_path=args.data_path,
                                     prompt_template=args.prompt_template)
    caption_dataloader = DataLoader(dataset=caption_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=8,
                                    pin_memory=True,
                                    shuffle=True)
    total_iters = len(caption_dataloader)

    for cur_iter, (filenames, prompts) in enumerate(caption_dataloader):
        from IPython import embed
        embed()
        print(f"Iteration: {cur_iter + 1}/{total_iters}")

    prompt = 'pass'

    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and Model
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument("--data-path", type=str, default="/DDN_ROOT/ytcheng/code/Open-VCLIP-V2/video_description_gen/back/caption_record.pth")
    # Hyper-parameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--prompt-template', type=str, required=True)
    # Devices
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "xpu"], default="cuda", help="The device type")
    parser.add_argument("--gpus", type=str, default=None, help="A single GPU like 1 or multiple GPUs like 0,2")
    parser.add_argument("--num-gpus", type=int, default=1)
    # LLM
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # Others
    parser.add_argument("--revision", type=str, default="main",help="Hugging Face Hub model revision identifier")
    parser.add_argument("--max-gpu-memory", type=str, help="The maximum memory per gpu. Use a string like '13Gib'")
    parser.add_argument("--load-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--cpu-offloading", action="store_true", help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU")
    parser.add_argument("--gptq-ckpt", type=str, default=None, help="Load quantized model. The path to the local GPTQ checkpoint.")
    parser.add_argument("--gptq-wbits", type=int, default=16, choices=[2, 3, 4, 8, 16], help="#bits to use for quantization")
    parser.add_argument("--gptq-groupsize", type=int, default=-1, help="Groupsize to use for quantization; default uses full row.")
    parser.add_argument("--gptq-act-order", action="store_true", help="Whether to apply the activation order GPTQ heuristic")

    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
