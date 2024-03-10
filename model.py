import torch
from data_prep import DataPreprocessor


class CharTrigramModel:
    def __init__(self, seed: int = 2147483647):
        self.seed = seed
        self.weights = torch.randn((729, 27), generator=torch.Generator().manual_seed(self.seed), requires_grad=True)

    def eval(self, xs: torch.Tensor, ys: torch.Tensor):
        with torch.no_grad():
            logits = self.weights[xs]
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)

            loss = -probs[torch.arange(xs.nelement()), ys].log().mean()

        return loss.item()

    def train(self,
              train_data: tuple[torch.Tensor, torch.Tensor],
              val_data: tuple[torch.Tensor, torch.Tensor],
              reg_factor: float | None = 0.0550,
              epochs: int | None = 150):
        train_xs, train_ys = train_data
        val_xs, val_ys = val_data
        for i in range(epochs):
            # forward pass
            logits = self.weights[train_xs]
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)
            regularization = reg_factor * (self.weights ** 2).mean()
            loss = -probs[torch.arange(train_xs.nelement()), train_ys].log().mean() + regularization

            train_loss = loss.item()
            val_loss = self.eval(val_xs, val_ys)

            print("EPOCH {}: LOSS train {} valid {}".format(i, train_loss, val_loss))

            # backward pass
            self.weights.grad = None
            loss.backward()
            with torch.no_grad():
                self.weights.data += -75 * self.weights.grad

    def sample(self,
               data_class: DataPreprocessor,
               seed: int | None = None,
               max_len: int | None = None,
               max_words: int | None = 1):
        if not seed:
            seed = self.seed

        g = torch.Generator().manual_seed(seed)

        out = []
        for i in range(max_words):
            ix = 1
            n = 0
            out.append(["."])
            while True:
                if ix != 1:
                    ix = data_class.char2_to_int[''.join(out[i][-2:])]
                with torch.no_grad():
                    logits = self.weights[ix]
                    counts = logits.exp()
                    probs = counts / counts.sum()

                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                out[i].append(data_class.int_to_char[ix])
                if max_len:
                    n += 1

                if max_len and n == max_len:
                    out[i].append(".")
                    out[i] = ''.join(out[i])
                    break

                if out[i][-1][-1] == ".":
                    out[i] = ''.join(out[i])
                    break

        return out
