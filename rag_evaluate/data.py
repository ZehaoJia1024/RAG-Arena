import json
from pathlib import Path


def load_data(data_dir):
    data_dir = Path(data_dir)
    documents = []
    for doc_path in data_dir.glob("*.txt"):
        doc = doc_path.read_text(encoding="utf-8")
        documents.append(doc)

    QA = data_dir / "评估问题.json"
    with open(QA, "r", encoding="utf-8") as f:
        QA = json.load(f)
    return documents, QA


if __name__ == '__main__':
    load_data("../data")
