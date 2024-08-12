import argparse
from llava.model.builder import load_pretrained_model

from llava.utils import build_logger

logger = build_logger("merge_model", f"merge_model.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    
    parser.add_argument("--save-merged-model", action="store_true")
    parser.add_argument("--save-dir", type=str, default="merged_model")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", type=str, default="Vi-VLM/Vistral-V")
    
    args = parser.parse_args()
    
    model_path = args.model_path
    model_base = args.model_base
    model_name = args.model_name
    load_8bit = args.load_8bit
    load_4bit = args.load_4bit
    device = args.device
    use_flash_attn = args.use_flash_attn
    
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    if model_name is None:
        model_paths = model_path.split("/")
        if model_paths[-1].startswith('checkpoint-'):
            model_name = model_paths[-2] + "_" + model_paths[-1]
        else:
            model_name = model_paths[-1]
            
    logger.info(f"Loading the model {model_name}")
    
    # FIXME: Vistral-V is need to be fixed, add prefix 'llava-'
    if 'vistral-v' in model_name.lower():
        model_name = 'llava-' + model_name
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit, load_4bit, device=device, use_flash_attn=use_flash_attn)
    
    if args.save_merged_model:
        tokenizer.save_pretrained(args.save_dir)
        model.save_pretrained(args.save_dir)
        logger.info(f"Model and tokenizer are saved to {args.save_dir}")
    if args.push_to_hub:
        tokenizer.push_to_hub(args.repo_id)
        model.push_to_hub(args.repo_id)
        logger.info(f"Model and tokenizer are pushed to {args.repo_id}")