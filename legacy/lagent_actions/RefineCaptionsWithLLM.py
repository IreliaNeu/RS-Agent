import os
import openai
import json
from typing import Annotated, List
from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode
from lagent.llms.openai import GPTAPI


# Deepseek_key = REDACTED_API_KEY
opanai_key = os.getenv("OPENAI_API_KEY", "")
model_name = 'gpt-3.5-turbo'


class RefineCaptionsWithLLM(BaseAction):
    """使用 GPTAPI 封装，对原始变化描述进行多样化增强。"""

    @tool_api
    def run(
        self,
        input_dir: Annotated[str, "包含原始描述的 .txt 文件目录"],
        output_dir: Annotated[str, "用于保存增强描述的新目录"],
        n_versions: Annotated[int, "每条描述生成多少个多样化版本"] = 3,
        model_name: Annotated[str, "使用的 GPT 模型名称"] = "gpt-3.5-turbo",
        output_format: Annotated[str, "输出格式：json（默认）或 list-token"] = "list-token",
        api_key: Annotated[str, "API key，例如 sk-xxx 或 openrouter 的 key"] = "",
        api_url: Annotated[str, "API base URL，可选。为空则使用默认"] = ""
    ) -> str:
        """
        可以调用python_interpreter工具生成可执行的函数，函数模版如下：
        def solution():
            from tools import RefineCaptionsWithLLM

            input_dir = 'xxxxxxx'
            output_dir = 'xxxxxxx'
            output_format = 'xxxxxxx'
            tool = RefineCaptionsWithLLM()
            result = tool.run(input_dir=input_dir, output_dir=output_dir, n_versions=3,output_format=output_format)
            return result
        Args:
            input_dir (str): 原始描述目录，每个 .txt 文件对应一个图像对描述（每文件首行）
            output_dir (str): 输出目录，每个文件保存为 .json 格式（包含 original 和 augmented 列表）
            n_versions (int): 每个输入生成的多样化版本数量
            output_format (str): 输出格式，json 或 list-token
            model_name (str): GPT 模型名称（如 gpt-3.5-turbo）

        Returns:
            str: JSON 格式的执行摘要
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        errors = []

        gpt = GPTAPI(model_type=model_name, key=api_key, openai_api_base=api_url)

        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith(".txt"):
                continue

            in_path = os.path.join(input_dir, fname)
            with open(in_path, 'r', encoding='utf-8') as f:
                original = f.readline().strip()

            if not original:
                errors.append(f"{fname}: empty input")
                continue

            prompt = (
                f"Please rewrite the following change description and generate {n_versions} versions with different styles, diverse language expressions, but consistent semantics,"
                f"Used for remote sensing image change detection tasks: \n\n\"{original}\"\n\n "
                f"The format of the output file is {output_format}"
                f"Please return all versions in the form of a list."
            )

            try:
                response_text = gpt.chat([{"role": "user", "content": prompt}])

                # 尝试解析成列表
                try:
                    if response_text.strip().startswith("["):
                        augmented = json.loads(response_text)
                        
                    else:
                        # 简单处理中文列表项
                        lines = response_text.split("\n")
                        augmented = [line.lstrip("1234567890. ").strip() for line in lines if line.strip()]
                except Exception:
                    augmented = [response_text]

                # output format selection: list-token.
                if output_format == "list-token":
                    formatted = [["<START>"] + sent.strip().split() + ["<END>"] for sent in augmented]
                    out_data = formatted
                    # out_path = os.path.join(output_dir, fname.replace('.txt', '_Multi.txt'))

                # output format selection: json.    

                else:
                    out_data = {
                        "original": original,
                        "augmented": augmented
                    }

                out_path = os.path.join(output_dir, fname.replace('.txt', '.json'))
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(out_data, f, ensure_ascii=False, indent=2)

                results.append({
                    "file": fname,
                    "augmented_count": len(augmented),
                    "output_format": output_format
                })

            except Exception as e:
                errors.append(f"{fname}: {str(e)}")

        return json.dumps({
            "total": len(results) + len(errors),
            "success": len(results),
            "failed": len(errors),
            "errors": errors
        }, indent=2, ensure_ascii=False)