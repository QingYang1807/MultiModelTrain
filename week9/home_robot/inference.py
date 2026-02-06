"""
推理脚本 - 家居机器人
"""

import os
import torch
from PIL import Image
from pathlib import Path
import gradio as gr
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="家居机器人推理")

    parser.add_argument(
        "--image", type=str, default=None,
        help="输入图像路径"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["grounding", "counting", "vqa", "all"],
        help="推理模式"
    )
    parser.add_argument(
        "--text", type=str, default="chair . sofa . table",
        help="文本提示"
    )
    parser.add_argument(
        "--question", type=str,
        default="What furniture do you see in this room?",
        help="VQA 问题"
    )
    parser.add_argument(
        "--category", type=str, default="chair",
        help="计数的类别"
    )
    parser.add_argument(
        "--output", type=str, default="output.jpg",
        help="输出图像路径"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="启动 Gradio 演示"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="创建共享的 Gradio 链接"
    )

    return parser.parse_args()


def demo_mode():
    """演示模式 - 使用占位图测试"""
    from model import create_home_robot

    print("\n=== 家居机器人演示 ===\n")

    # 创建机器人
    robot = create_home_robot()

    # 创建测试图像
    test_image = Image.new("RGB", (640, 480), color=(180, 180, 180))
    test_image_path = "demo_image.jpg"
    test_image.save(test_image_path)

    print(f"测试图像: {test_image_path}\n")

    # Grounding 测试
    print("--- Grounding 测试 ---")
    if robot.grounding_module:
        results = robot.ground("chair . sofa . table . bed", test_image_path)
        print(f"检测到 {len(results)} 个目标")
        for r in results:
            print(f"  - {r}")
    else:
        print("Grounding 模块不可用")

    print()

    # Counting 测试
    print("--- Counting 测试 ---")
    if robot.counting_module:
        for cat in ["chair", "table", "bed"]:
            count = robot.count(cat, test_image_path)
            print(f"{cat}: {count}")
    else:
        print("Counting 模块不可用")

    print()

    # VQA 测试
    print("--- VQA 测试 ---")
    if robot.vqa_module:
        questions = [
            "What furniture do you see?",
            "How many chairs are there?",
            "Describe the room.",
        ]
        for q in questions:
            answer = robot.vqa(q, test_image_path)
            print(f"Q: {q}")
            print(f"A: {answer}\n")
    else:
        print("VQA 模块不可用")

    print("演示完成！")


def gradio_demo():
    """Gradio 演示界面"""
    from model import create_home_robot

    # 创建机器人
    robot = create_home_robot()

    def process_image(
        image,
        text_prompt,
        question,
        category,
        mode,
    ):
        """处理图像"""
        if image is None:
            return "请上传图像", None, {}

        results = {}

        if mode in ["grounding", "all"]:
            if robot.grounding_module:
                detections = robot.ground(text_prompt, image)
                results["detections"] = detections

        if mode in ["counting", "all"]:
            if robot.counting_module:
                count = robot.count(category, image)
                results["count"] = count

        if mode in ["vqa", "all"]:
            if robot.vqa_module:
                answer = robot.vqa(question, image)
                results["answer"] = answer

        # 格式化输出
        output_text = []
        if "detections" in results:
            output_text.append(
                f"检测到 {len(results['detections'])} 个目标"
            )
        if "count" in results:
            output_text.append(f"{category}: {results['count']}")
        if "answer" in results:
            output_text.append(f"回答: {results['answer']}")

        return "\n".join(output_text), image, results

    # 构建界面
    with gr.Blocks(title="家居机器人") as demo:
        gr.Markdown("# 家居机器人 - Grounding + Counting + VQA")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="上传图像",
                )

                mode_radio = gr.Radio(
                    ["grounding", "counting", "vqa", "all"],
                    label="选择模式",
                    value="all",
                )

                with gr.Group(visible=True) as grounding_group:
                    text_prompt = gr.Textbox(
                        label="文本提示",
                        value="chair . sofa . table . bed",
                    )

                with gr.Group(visible=True) as counting_group:
                    category = gr.Textbox(
                        label="计数类别",
                        value="chair",
                    )

                with gr.Group(visible=True) as vqa_group:
                    question = gr.Textbox(
                        label="VQA 问题",
                        value="What furniture do you see?",
                    )

                submit_btn = gr.Button("执行分析", variant="primary")

            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="分析结果",
                    lines=10,
                )
                output_image = gr.Image(
                    type="pil",
                    label="可视化结果",
                )

        # 绑定事件
        submit_btn.click(
            process_image,
            inputs=[
                image_input, text_prompt, question,
                category, mode_radio,
            ],
            outputs=[output_text, output_image],
        )

    return demo


def main():
    """主函数"""
    args = parse_args()

    if args.demo:
        # 启动 Gradio 演示
        demo = gradio_demo()
        demo.launch(share=args.share)
    elif args.image:
        # 单张图像推理
        from model import create_home_robot

        robot = create_home_robot()

        print(f"\n=== 家居机器人 - 单张图像推理 ===")
        print(f"图像: {args.image}")
        print(f"模式: {args.mode}\n")

        if args.mode in ["grounding", "all"]:
            print("--- Grounding ---")
            if robot.grounding_module:
                results = robot.ground(args.text, args.image)
                print(f"检测到 {len(results)} 个目标")
                for r in results:
                    print(f"  - {r}")

        if args.mode in ["counting", "all"]:
            print("\n--- Counting ---")
            if robot.counting_module:
                count = robot.count(args.category, args.image)
                print(f"{args.category}: {count}")

        if args.mode in ["vqa", "all"]:
            print("\n--- VQA ---")
            if robot.vqa_module:
                answer = robot.vqa(args.question, args.image)
                print(f"Q: {args.question}")
                print(f"A: {answer}")

    else:
        # 默认演示
        demo_mode()


if __name__ == "__main__":
    main()
