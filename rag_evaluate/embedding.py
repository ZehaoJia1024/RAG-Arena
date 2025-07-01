import os
from typing import Union, List
from sentence_transformers import SentenceTransformer
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Embedder:
    """
    一个集成了文本向量化、相关性计算和Top-K检索的RAG工具。
    get_top_k 方法使用高效的矩阵运算来处理批量查询。
    """

    def __init__(self):
        self.model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    def embed(
            self,
            texts: Union[str, List[str]],
    ) -> np.ndarray:
        is_single_string = isinstance(texts, str)
        input_texts = [texts] if is_single_string else texts
        embeddings = self.model.encode(input_texts)
        return embeddings[0] if is_single_string else embeddings

    def similarity(self, query_embeddings, document_embeddings):
        return self.model.similarity(query_embeddings, document_embeddings).cpu().numpy()

    @staticmethod
    def get_top_k(
            similarity_matrix,
            k: int = 3,
    ) -> Union[List[dict], List[List[dict]]]:
        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k:][:, ::-1]
        return top_k_indices.tolist()



