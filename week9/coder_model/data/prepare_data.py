"""
数据准备脚本
"""

import os
import json
import requests
from pathlib import Path


def download_code_alpaca(data_dir: str, max_samples: int = None):
    """
    下载 Code Alpaca 数据集

    Args:
        data_dir: 数据保存目录
        max_samples: 最大样本数，None 表示全部
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 下载训练数据
    train_url = (
        "https://raw.githubusercontent.com/"
        "sahil280114/codealpaca/main/data/code_alpaca_20k.json"
    )

    print(f"下载 Code Alpaca 训练数据...")
    response = requests.get(train_url)
    response.raise_for_status()

    data = response.json()

    # 限制样本数
    if max_samples:
        data = data[:max_samples]

    # 保存数据
    train_path = data_dir / "code_alpaca_train.json"
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"训练数据已保存到: {train_path}")
    print(f"总样本数: {len(data)}")

    # 划分验证集
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    # 保存验证集
    eval_path = data_dir / "code_alpaca_eval.json"
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"验证数据已保存到: {eval_path}")
    print(f"训练样本: {len(train_data)}, 验证样本: {len(eval_data)}")

    return train_data, eval_data


def create_sample_data(data_dir: str, num_samples: int = 100):
    """
    创建示例数据（用于测试）

    Args:
        data_dir: 数据保存目录
        num_samples: 示例数量
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 示例代码生成任务
    sample_tasks = [
        {
            "instruction": "Write a Python function to calculate the factorial of a number.",
            "input": "n = 5",
            "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\n# Example usage\nprint(factorial(5))  # Output: 120"
        },
        {
            "instruction": "Write a Python function to check if a string is a palindrome.",
            "input": "s = 'racecar'",
            "output": "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]\n\n# Example usage\nprint(is_palindrome('racecar'))  # Output: True"
        },
        {
            "instruction": "Write a Python function to find the maximum element in a list.",
            "input": "arr = [3, 1, 4, 1, 5, 9, 2, 6]",
            "output": "def find_max(arr):\n    if not arr:\n        return None\n    max_val = arr[0]\n    for num in arr:\n        if num > max_val:\n            max_val = num\n    return max_val"
        },
        {
            "instruction": "Write a Python function to sort a list using bubble sort.",
            "input": "arr = [64, 34, 25, 12, 22, 11, 90]",
            "output": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"
        },
        {
            "instruction": "Write a Python function to check if two strings are anagrams.",
            "input": "s1 = 'listen', s2 = 'silent'",
            "output": "def are_anagrams(s1, s2):\n    return sorted(s1) == sorted(s2)\n\n# Example usage\nprint(are_anagrams('listen', 'silent'))  # Output: True"
        },
    ]

    # 生成更多变体
    import random
    import string

    data = []
    for i in range(num_samples):
        template = random.choice(sample_tasks)
        data.append({
            "instruction": template["instruction"],
            "input": template["input"],
            "output": template["output"],
        })

    # 保存
    train_path = data_dir / "code_alpaca_train.json"
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"示例数据已保存到: {train_path}")
    print(f"样本数: {len(data)}")

    # 划分验证集
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    # 保存验证集
    eval_path = data_dir / "code_alpaca_eval.json"
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"验证数据已保存到: {eval_path}")
    print(f"训练样本: {len(train_data)}, 验证样本: {len(eval_data)}")

    return data


def format_data(data: list, template: dict = None) -> list:
    """
    格式化数据为训练格式

    Args:
        data: 原始数据列表
        template: 模板配置

    Returns:
        格式化后的数据列表
    """
    if template is None:
        template = {
            "prompt": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
            "response": "{output}",
        }

    formatted_data = []
    for item in data:
        prompt = template["prompt"].format(
            instruction=item.get("instruction", ""),
            input=item.get("input", ""),
        )
        response = template["response"].format(
            output=item.get("output", ""),
        )

        formatted_data.append({
            "prompt": prompt,
            "response": response,
            "full_text": prompt + response,
        })

    return formatted_data


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="准备 Code Alpaca 数据")
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="数据保存目录"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="最大样本数"
    )
    parser.add_argument(
        "--create_sample", action="store_true",
        help="创建示例数据"
    )

    args = parser.parse_args()

    if args.create_sample:
        print("创建示例数据...")
        create_sample_data(args.data_dir, num_samples=100)
    else:
        print("下载 Code Alpaca 数据集...")
        download_code_alpaca(args.data_dir, max_samples=args.max_samples)

    # 格式化数据
    template = {
        "prompt": "Below is an instruction that describes a task. "
                  "Write a response that appropriately completes the request.\n\n"
                  "### Instruction:\n{instruction}\n\n"
                  "### Input:\n{input}\n\n"
                  "### Response:",
        "response": "{output}",
    }

    # 加载并格式化
    train_path = os.path.join(args.data_dir, "code_alpaca_train.json")
    with open(train_path, 'r') as f:
        data = json.load(f)

    formatted = format_data(data, template)

    # 保存格式化后的数据
    formatted_path = os.path.join(args.data_dir, "code_alpaca_formatted.json")
    with open(formatted_path, 'w') as f:
        json.dump(formatted, f, indent=2)

    print(f"格式化数据已保存到: {formatted_path}")


if __name__ == "__main__":
    main()
