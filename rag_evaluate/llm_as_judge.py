import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI


class LLMJudge:
    """
    一个使用大语言模型（LLM）作为评委来评估RAG系统回答质量的类。

    它通过构建一个包含问题、上下文和答案的Prompt，
    然后调用LLM获取结构化的评估分数和理由。
    """

    # 一个通用且强大的默认系统Prompt，指导LLM进行评估
    DEFAULT_SYSTEM_PROMPT = """
    # 角色
    你是一个用于评估检索增强生成（RAG）系统输出质量的专家级AI评判员。你的评估必须严格、客观，并遵循以下所有指令。

    # 任务
    你的评估必须基于以下三个输入：
    1. 【用户问题】
    2. 【正确答案】 (Ground Truth，作为黄金标准)
    3. 【生成答案】 (需要被评估的答案)

    请根据以下三个维度进行评估，并为每个维度打出1-5分的整数分数：

    1.  **Correctness (正确性/语义等价性)**:
        - 评估【生成答案】的核心意思是否与【正确答案】完全一致。
        - 5分: 语义上完全等价，是【正确答案】的完美复述或高质量概括。
        - 3分: 抓住了【正确答案】的主要观点，但丢失了一些次要细节或表述略有偏差。
        - 1分: 与【正确答案】的核心观点相悖，或包含了严重的错误信息。

    2.  **Completeness (完整性)**:
        - 评估【生成答案】是否包含了【正确答案】中的所有关键信息点。
        - 5分: 完整覆盖了【正确答案】中的所有要点，没有遗漏。
        - 3分: 遗漏了【正确答案】中的一些次要信息点。
        - 1分: 遗漏了【正确答案】中的核心关键信息。
        
    3.  **Clarity & Conciseness (清晰简洁性)**:
        - 评估【生成答案】的语言质量，是否流畅、易懂、无冗余。
        - 5分: 语言表达清晰、准确、简洁，逻辑性强。
        - 3分: 语言基本通顺，但略显冗长或结构有些混乱。
        - 1分: 语言表达晦涩，难以理解，或充满了无意义的重复。

    【输出格式要求】
    你的评估结果必须以一个严格的JSON对象格式返回，包含 `scores` (内含三个维度分数), `reasoning` (总体文字说明)。
    【输出格式示例】
    ```json
    {
      "scores": {
        "Correctness": 5,
        "Completeness": 5,
        "Clarity & Conciseness": 4,
      },
      "reasoning": "答案忠实于原文，并精准回答了问题。上下文文档基本相关，但有一篇相关性稍弱。答案遗漏了一个次要细节，但整体表达清晰简洁。",
    }
    ```
    """

    def __init__(
            self,
            model: str = "qwen3-235b-a22b",
            system_prompt: Optional[str] = None
    ):
        """
        初始化LLMJudge。

        Args:
            model (str): 要使用的LLM评委模型名称。
            system_prompt (Optional[str]): 自定义的系统提示词。如果为None，则使用默认提示词。
        """
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _create_eval_prompt(
            self, query: str, generated_answer: str, ground_truth_answer: str
    ) -> str:
        return f"""
        请根据你在系统指令中得到的评估标准，对以下内容进行评估：

        【用户问题】:
        {query}

        【正确答案】(Ground Truth):
        {ground_truth_answer}

        【生成答案】(To be evaluated):
        {generated_answer}

        请严格按照JSON格式输出你的评估结果。
        """

    def judge_answer(
            self,
            query: str,
            ground_truth_answer: str,
            generated_answer: str,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """
        调用LLM对给定的答案进行评估。

        Args:
            query (str): 用户的原始问题。
            ground_truth_answer (str): 正确答案。
            generated_answer (str): RAG系统生成的最终答案。
            **kwargs: 其他可以传递给 `client.chat.completions.create` 的参数，
                      例如 temperature=0.0, max_tokens=500。

        Returns:
            Dict[str, Any]: 一个包含评估结果的字典。
        """
        user_prompt = self._create_eval_prompt(query, generated_answer, ground_truth_answer)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 为API调用设置默认参数，允许用户通过kwargs覆盖
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,  # 评估任务需要确定性，所以温度设为0
            "max_tokens": 1024,
            "stream": True,
            "extra_body": {"enable_thinking": True, "top_k": 1}
        }

        response = self.client.chat.completions.create(**api_params)
        raw_output = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if not content:
                continue
            raw_output += content

        # 有时LLM会返回被Markdown代码块包裹的JSON，需要先去除
        if raw_output.strip().startswith("```json"):
            raw_output = raw_output.strip()[7:-3].strip()

        evaluation = json.loads(raw_output)
        evaluation["final_score"] = sum(evaluation["scores"].values()) / 3
        return evaluation

    def evaluate(self, queries, answers_gt, answers):
        total_score = 0

        for i, (query, answer_gt, answer) in enumerate(zip(queries, answers_gt, answers), start=1):
            print(f"{i} 正在评估问题：{query}")
            print('-' * 20)
            evaluation = self.judge_answer(query, answer, answer_gt)
            total_score += evaluation["final_score"]
            print(evaluation)
            print("=" * 20)
        print(f"最终得分：{total_score / len(queries):.2f}")