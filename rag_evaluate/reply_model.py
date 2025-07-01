import os
import numpy as np
from typing import List
from openai import OpenAI



class ReplyModel:
    """
    一个最核心的RAG回复模型。
    它在接收到问题时，即时执行检索和生成两个步骤。
    """
    SYSTEM_PROMPT = """
    你是一个问答机器人。请严格根据下面提供的“参考文档”来回答问题。
    如果文档中的信息不足以回答，直接回复“根据提供的文档，我无法回答该问题。”
    """

    def __init__(
            self,
            model_name: str = "qwen3-4b"
    ):
        self.model_name = model_name

        # 初始化用于生成答案的LLM客户端
        self.llm_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def answer(self, query: str, retrieved_docs: list[str]) -> str:
        # 构建上下文和Prompt
        context_str = "\n\n".join(retrieved_docs)
        user_prompt = f"""
        参考文档:
        ---
        {context_str}
        ---

        问题: {query}
        """

        # 调用LLM生成答案
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        completion = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            seed=0,
            extra_body={"enable_thinking": False}
        )

        # 返回最终的答案文本
        return completion.choices[0].message.content
