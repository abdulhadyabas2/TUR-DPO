import json

pairs = [
    {
        "prompt": "What is 2 + 3? Explain.",
        "chosen": "We add two and three to get five. Therefore, the answer is 5.",
        "rejected": "Two plus three equals six because I thought of 3+3.",
        "ref_answer": "5",
    },
    {
        "prompt": "Summarize: Cats are animals. They like naps.",
        "chosen": "Cats are animals that enjoy naps.",
        "rejected": "Dogs are animals that enjoy naps.",
        "ref_answer": "cats enjoy naps",
    },
]

with open("data/sample_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

print("Wrote data/sample_pairs.jsonl with", len(pairs), "lines")
