"""
推理脚本 - Code Alpaca LoRA 微调
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Code Alpaca 推理")

    parser.add_argument(
        "--base_model", type=str,
        default="TinyPixel/Llama-2-7B-bf16-sharded",
        help="基础模型名称"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="微调后的模型路径"
    )
    parser.add_argument(
        "--input_file", type=str, default="data/code_alpaca_eval.json",
        help="输入数据文件"
    )
    parser.add_argument(
        "--output_file", type=str, default="outputs/predictions.json",
        help="输出结果文件"
    )
    parser.add_argument(
        "--compare_base", action="store_true",
        help="对比微调前后的效果"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="最大生成长度"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="温度"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10,
        help="测试样本数量"
    )
    parser.add_argument(
        "--test_instruction", type=str, default=None,
        help="测试指令"
    )

    return parser.parse_args()


def load_model(base_model: str, model_path: str = None):
    """
    加载模型

    Args:
        base_model: 基础模型名称
        model_path: 微调后的模型路径

    Returns:
        模型和分词器
    """
    print(f"加载基础模型: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载微调后的模型
    if model_path and os.path.exists(model_path):
        print(f"加载微调模型: {model_path}")
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            ),
            model_path,
        )
        is_finetuned = True
    else:
        print("使用基础模型")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        is_finetuned = False

    return model, tokenizer, is_finetuned


def generate(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
):
    """
    生成代码

    Args:
        model: 模型
        tokenizer: 分词器
        instruction: 指令
        input_text: 输入
        max_new_tokens: 最大生成长度
        temperature: 温度

    Returns:
        生成的代码
    """
    # 构建提示
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""

    # 分词
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 解码
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    # 提取生成部分
    if prompt in generated_text:
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()

    return response


def evaluate_model(
    model,
    tokenizer,
    test_data: list,
    max_new_tokens: int = 256,
):
    """
    评估模型

    Args:
        model: 模型
        tokenizer: 分词器
        test_data: 测试数据

    Returns:
        评估结果列表
    """
    results = []

    for item in tqdm(test_data, desc="评估"):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        expected = item.get("output", "")

        # 生成
        generated = generate(
            model, tokenizer,
            instruction, input_text,
            max_new_tokens=max_new_tokens,
        )

        results.append({
            "instruction": instruction,
            "input": input_text,
            "expected": expected,
            "generated": generated,
        })

    return results


def main():
    """主函数"""
    args = parse_args()

    # 加载测试数据
    with open(args.input_file, 'r') as f:
        test_data = json.load(f)

    test_data = test_data[:args.num_samples]

    if args.test_instruction:
        # 单个测试
        print("\n=== 单个测试 ===")
        print(f"指令: {args.test_instruction}")

        # 加载模型
        model, tokenizer, is_finetuned = load_model(
            args.base_model,
            args.model_path,
        )

        # 生成
        response = generate(
            model, tokenizer,
            args.test_instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print(f"\n生成结果:\n{response}")

    elif args.compare_base:
        # 对比微调前后
        print("\n=== 对比微调前后效果 ===")

        # 加载基础模型
        print("\n1. 基础模型结果:")
        base_model, tokenizer, _ = load_model(
            args.base_model,
            model_path=None,
        )

        base_results = evaluate_model(
            base_model, tokenizer, test_data,
            max_new_tokens=args.max_new_tokens,
        )

        # 加载微调模型
        print("\n2. 微调模型结果:")
        finetuned_model, tokenizer, _ = load_model(
            args.base_model,
            model_path=args.model_path,
        )

        finetuned_results = evaluate_model(
            finetuned_model, tokenizer, test_data,
            max_new_tokens=args.max_new_tokens,
        )

        # 保存结果
        output = {
            "base_model": args.base_model,
            "finetuned_model": args.model_path,
            "num_samples": len(test_data),
            "base_results": base_results[:5],  # 只保存前5个
            "finetuned_results": finetuned_results[:5],
        }

        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {args.output_file}")

        # 打印对比示例
        print("\n=== 对比示例 ===")
        for i, (base, finetuned) in enumerate(
            zip(base_results[:3], finetuned_results[:3])
        ):
            print(f"\n--- 样本 {i + 1} ---")
            print(f"指令: {base['instruction'][:100]}...")
            print(f"基础模型:\n{base['generated'][:200]}...")
            print(f"微调模型:\n{finetuned['generated'][:200]}...")

    else:
        # 普通推理
        print("\n=== 代码生成推理 ===")

        # 加载模型
        model, tokenizer, is_finetuned = load_model(
            args.base_model,
            args.model_path,
        )

        print(f"使用微调模型: {is_finetuned}")

        # 生成测试
        for item in test_data[:5]:
            print(f"\n指令: {item['instruction'][:80]}...")
            print(f"输入: {item['input'][:50] if item['input'] else '无'}")

            response = generate(
                model, tokenizer,
                item['instruction'],
                item.get('input', ''),
                max_new_tokens=args.max_new_tokens,
            )

            print(f"生成:\n{response[:200]}...")
            print("-" * 50)


if __name__ == "__main__":
    main()
