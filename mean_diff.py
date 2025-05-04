import torch

acts = torch.load("data/gemma_prompt_activations.pt")
X, y = acts["activations"], acts["labels"]
pos = X[y == target_class]     
neg = X[y != target_class]

direction = (pos.mean(0) - neg.mean(0)).float().cuda()
direction /= direction.norm()
torch.save(direction.cpu(), "concept_vectors/hallucination_mean_diff.pt")
