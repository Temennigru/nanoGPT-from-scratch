from model import GPT
import torch
from nltk.corpus import gutenberg
import tiktoken
import inspect
from tqdm import tqdm
import sys
import time
import os
import numpy as np


class Logger:
    def __init__(self):
        self.progress = None
        self.max_steps = None
        self.step = None

    def write(self, text):
        if self.progress is None:
            tqdm.write(text)
        else:
            tqdm.write("\r" + text)

    def create_bar(self, max_steps):
        sys.stdout.flush()
        time.sleep(0.1) # Give the system time to flush
        self.max_steps = max_steps
        self.progress = tqdm(total=max_steps, unit="step")
        self.step = 0
        self.progress.set_description(f"Training step {self.step}/{max_steps}")

    def step_bar(self):
        self.progress.update(1)
        self.step += 1
        self.progress.set_description(f"Training step {self.step}/{self.max_steps}")

    def close_bar(self):
        sys.stdout.flush()
        time.sleep(0.1) # Give the system time to flush
        self.progress.update()
        self.progress.close()
        self.progress = None
        self.max_steps = None


class MyEncoder:
    def __init__(self, text):
        self.vocab = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}
        self.n_vocab = len(self.vocab)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, encoded):
        return "".join([self.itos[i] for i in encoded])


class Trainer:
    def __init__(self,
                 data_dir,
                 encoder,
                 model,
                 seed,
                 gpu=False):
        #data = open("shakespeare.txt").read()
        #n = int(training_split * len(data))
        #self.train_data = torch.tensor(self.encoder.encode(data[:n]), dtype=torch.long)
        #self.val_data = torch.tensor(self.encoder.encode(data[n:]), dtype=torch.long)

        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        self.logger = Logger()
        self.log(f"train data: {len(self.train_data)}")
        self.log(f"val data: {len(self.val_data)}")
        self.encoder = encoder
        self.block_size = model.block_size

        self.cuda = torch.cuda.is_available() and gpu
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        model.device = self.device
        self.model = model.to(self.device)

        torch.manual_seed(seed)

    def log(self, text):
        self.logger.write(text)

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [par for name, par in param_dict.items() if par.dim() >= 2]
        nodecay_params = [par for name, par in param_dict.items() if par.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        self.log(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        self.log(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.cuda
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        self.log(f"using fused AdamW: {use_fused}")

        return optimizer

    def get_batch(self, split, batch_size):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        #x = torch.stack([data[i:i + self.block_size] for i in ix])
        #y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])

        if self.cuda:
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = self.get_batch(split, batch_size=32)
                logits, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self,
              steps,
              batch_size,
              learning_rate,
              weight_decay,
              beta1,
              beta2,
              eval_interval,
              eval_iters):
        #optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        optimizer = self.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))

        self.logger.create_bar(steps)
        for step in range(steps):
            # Sample batch
            xb, yb = self.get_batch('train', batch_size)

            # evaluate loss
            out, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % eval_interval == 0 and step < steps-1:
                losses = self.estimate_loss(eval_iters)
                self.log(f"step {step + 1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            self.logger.step_bar()

        losses = self.estimate_loss(eval_iters)
        self.log(f"step {step + 1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        self.logger.close_bar()
        self.log("Training finished")

    def generate(self, batch_size=4, max_tokens=100, temperature=1.0, top_k=None, text=None):
        # xb, yb = self.get_batch('validate', batch_size)

        if text is None:
            idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        else:
            idx = (torch.tensor(self.encoder.encode(text), dtype=torch.long, device=self.device)[None, ...])

        gen = self.model.generate(idx, max_tokens, temperature, top_k)

        out = gen[0].tolist()

        self.log(self.encoder.decode(out))


def gutenberg_shakespeare():
    corpus = [i for i in gutenberg.fileids() if 'shakespeare' in i]
    text = ""

    for i in corpus:
        text += gutenberg.raw(i) + "\n\n"

    return text


def main():

    encoder = tiktoken.get_encoding('gpt2')
    #encoder = MyEncoder(text)

    lm = GPT(
        encoder.n_vocab,
        block_size=256,
        head_size=64,
        n_heads=8,
        n_blocks=8,
        dropout=0.1)

    trainer = Trainer(
        "D:\\ML\\nanoGPT-master\\data\\openwebtext",
        encoder, lm, gpu=True, seed=1337)

    trainer.train(
        steps=10000,
        eval_interval=500,
        eval_iters=100,
        batch_size=48,
        learning_rate=6e-4,
        weight_decay=1e-1,
        beta1=0.9,
        beta2=0.95,
    )

    trainer.generate(text="hello, what is your name?", max_tokens=10000, temperature=0.8, top_k=400)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()