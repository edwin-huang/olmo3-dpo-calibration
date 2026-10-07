"""List Olmo 3 7B repos and their checkpoint branches on Hugging Face.

Run this first: exact repo names and branch names change between releases,
so we read them from the Hub instead of hardcoding guesses.
"""
from huggingface_hub import HfApi

api = HfApi()

repos = sorted(
    m.id for m in api.list_models(author="allenai", search="Olmo-3")
    if "7B" in m.id or "7b" in m.id
)
print("== Olmo 3 7B repos ==")
for r in repos:
    print(r)

print("\n== Branches per repo (intermediate checkpoints) ==")
for r in repos:
    try:
        branches = [b.name for b in api.list_repo_refs(r).branches]
    except Exception as e:  # gated or private repos
        print(f"{r}: <error: {e}>")
        continue
    branches.sort()
    head = ", ".join(branches[:5])
    tail = ", ".join(branches[-5:]) if len(branches) > 10 else ""
    print(f"{r}: {len(branches)} branches  [{head}{' ... ' + tail if tail else ''}]")
