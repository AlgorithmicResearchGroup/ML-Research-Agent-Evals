import torch
import tiktoken
from torch.nn import functional as F
from contextlib import nullcontext
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model_Generate:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.max_new_tokens = 512
        self.temperature = 0.8
        self.top_k = 200
        self.block_size = 256
        
    def generation(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, logits

    def compute_logits(self, tokens):
        with torch.no_grad():
            logits, _ = self.model(tokens)
        return logits

    def return_response(self, start_ids):
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        with torch.no_grad():
            y, logits = self.generation(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
            text = self.tokenizer.decode(y[0].tolist())   
        return text, y, logits