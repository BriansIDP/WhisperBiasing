from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .transcribe import transcribe as transcribe_function
from .decoding import detect_language as detect_language_function, decode as decode_function
from .GNN import GCN


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits

    def get_states(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits, x


class WhisperBiasing(nn.Module):
    def __init__(self, whisper, tokenizer, embdim, hiddim, attndim, nvocab, Bdrop=0.0, biasing=False,
                 GNNtype="none", GNNdim=0, useGPT=False, GPThiddim=0):
        super().__init__()
        self.whisper = whisper
        self.tokenizer = tokenizer
        self.eos = self.tokenizer.tokenizer.eos_token_id
        self.treehid = embdim if GNNdim == 0 else GNNdim

        # GNN module
        self.GNNtype = GNNtype
        self.GNN = None
        if GNNtype != "none":
            # gcn3 means using GCN with 3 layers
            if GNNtype.startswith("gcn"):
                self.GNN = GCN(
                    embdim,
                    self.treehid,
                    int(self.GNNtype[3:]),
                    Bdrop,
                    residual=False,
                )
            lecun_normal_init_parameters(self.GNN)

        self.hiddim = hiddim
        self.attndim = attndim
        self.nvocab = nvocab
        self.useGPT = useGPT
        self.GPThiddim = GPThiddim
        if not useGPT:
            self.pointer_gate = torch.nn.Linear(self.attndim + self.hiddim, 1)
        else:
            self.pointer_gate = torch.nn.Linear(self.attndim + self.hiddim + self.GPThiddim, 3)
        self.Qproj = torch.nn.Linear(self.hiddim, self.attndim)
        self.Kproj = torch.nn.Linear(self.treehid, self.attndim)
        self.ooKBemb = torch.nn.Embedding(1, self.treehid)
        self.Bdrop = torch.nn.Dropout(Bdrop)
        self.biasing = biasing

        lecun_normal_init_parameters(self.Qproj)
        lecun_normal_init_parameters(self.Kproj)
        lecun_normal_init_parameters(self.pointer_gate)
        lecun_normal_init_parameters(self.ooKBemb)

    def get_step_biasing_embs_prefix(self, yseq, trees, origTries):
        ooKB_id = self.nvocab
        p_gen_mask = []
        maxlen = 0
        index_list = []
        new_trees = []
        masks_list = []
        step_embs = []
        for i, char_idx in enumerate(yseq):
            new_tree = trees[i][0]
            char_idx = char_idx.item()
            extended_list = []
            if char_idx == self.eos:
                new_tree = origTries[i].copy()
                index_list.append(list(new_tree[0].keys()))
            elif char_idx > 0 and self.tokenizer.decode([char_idx]).startswith(' '):
                new_tree =  origTries[i].copy()
                if char_idx not in new_tree[0]:
                    index_list.append(list(new_tree[0].keys()))
                else:
                    new_tree = new_tree[0][char_idx]
                    index_list.append(list(new_tree[0].keys()))
                    if new_tree[1] != -1:
                        index_list[-1].extend(list(origTries[i][0].keys()))
                        extended_list = list(origTries[i][0].keys())
            else:
                if char_idx not in new_tree:
                    new_tree = origTries[i].copy()
                    index_list.append(list(new_tree[0].keys()))
                else:
                    new_tree = new_tree[char_idx]
                    index_list.append(list(new_tree[0].keys()))
                    if new_tree[1] != -1:
                         index_list[-1].extend(list(origTries[i][0].keys()))
                         extended_list = list(origTries[i][0].keys())
            if char_idx > 0:
                p_gen_mask.append(0)
            else:
                p_gen_mask.append(1)
            new_trees.append(new_tree)
            if len(index_list[-1]) > maxlen:
                maxlen = len(index_list[-1])

            if getattr(self, "GNN", None) is not None:
                step_emb = [new_tree[0][key][3] for key in new_tree[0].keys()]
                if extended_list != []:
                    step_emb.extend([origTries[i][0][key][3] for key in extended_list])
                if len(step_emb) > 0:
                    step_embs.append(torch.cat(step_emb, dim=0))
                else:
                    step_embs.append(to_device(self, torch.empty(0, self.tree_hid)))

        maxlen += 1
        step_mask = []
        back_transform = torch.zeros(len(new_trees), maxlen, ooKB_id+1, device=yseq.device)
        ones_mat = torch.ones(back_transform.size(), device=yseq.device)
        for i, indices in enumerate(index_list):
            step_mask.append(len(indices) * [0] + (maxlen - len(indices) - 1) * [1] + [0])
            if getattr(self, "GNN", None) is not None:
                pad_embs = self.ooKBemb.weight.repeat(maxlen-len(indices), 1)
                step_embs[i] = torch.cat([step_embs[i], pad_embs], dim=0)
            indices += [ooKB_id] * (maxlen - len(indices))
        step_mask = torch.tensor(step_mask).byte().to(yseq.device)
        index_list = torch.LongTensor(index_list).to(yseq.device)
        back_transform.scatter_(dim=-1, index=index_list.unsqueeze(-1), src=ones_mat)
        if getattr(self, "GNN", None) is not None:
            step_embs = torch.stack(step_embs)

        return step_mask, step_embs, new_trees, p_gen_mask, back_transform, index_list

    def get_meetingKB_emb_map(
            self,
            query,
            meeting_mask,
            back_transform,
            index_list,
            meeting_KB=[],
        ):
        if getattr(self, "GNN", None) is None or meeting_KB == []:
            meeting_KB = torch.cat([self.whisper.decoder.token_embedding.weight.data, self.ooKBemb.weight], dim=0)
            meeting_KB = meeting_KB[index_list]
        meeting_KB = self.Bdrop(self.Kproj(meeting_KB))
        KBweight = torch.einsum('ijk,ik->ij', meeting_KB, query)
        KBweight = KBweight / math.sqrt(query.size(-1))
        KBweight.masked_fill_(meeting_mask.bool(), -1e9)
        KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
        if meeting_KB.size(1) > 1:
            KBembedding = torch.einsum('ijk,ij->ik', meeting_KB[:,:-1,:], KBweight[:,:-1])
        else:
            KBembedding = KBweight.new_zeros(meeting_KB.size(0), meeting_KB.size(-1))
        KBweight = torch.einsum('ijk,ij->ik', back_transform, KBweight)
        return KBembedding, KBweight

    def calc_ptr_loss(self, ptr_dist, model_dist, ptr_gen, ptr_gen_mask,
                      targets, ignore_idx=-100, reduction_str='none'):
        ptr_gen = ptr_gen.squeeze(-1).masked_fill(ptr_gen_mask.bool(), 0).reshape(-1, 1)
        # the gap to 1 is the prob for <unk>, which indicates not in the KB
        ptr_gen_complement = (ptr_dist[:,:,-1].reshape(targets.size(0), -1)) * ptr_gen
        # print((ptr_dist[:,:,:-1].reshape(targets.size(0), -1) * ptr_gen).sum(-1).max())
        p_final = ptr_dist[:,:,:-1].reshape(targets.size(0), -1) * ptr_gen + model_dist * (1 - ptr_gen + ptr_gen_complement)
        p_loss = F.nll_loss(torch.log(p_final+1e-9), targets,
                            ignore_index=ignore_idx, reduction=reduction_str)
        p_loss = p_loss.sum() / (p_loss != 0).sum()
        return p_loss, p_final

    def forward(self, fbank, targets, targetmask, lextree, sotlen, GPThidden=None):
        # Forwarding model
        with torch.no_grad():
            logits, hidden = self.whisper.getstates(fbank, targets * targetmask)

        if self.biasing:
            # Forward GNN first
            if getattr(self, "GNN", None) is not None:
                self.whisper.decoder.token_embedding.weight.requires_grad = False
                self.GNN(lextree, self.whisper.decoder.token_embedding)
            # Duplicate lextree
            lextrees = [lextree for i in range(targets.size(0))]

            hptrs = []
            tcpgen_dists = []
            p_gen_masks = []
            trees = lextrees
            query = self.Bdrop(self.Qproj(hidden))
            for i in range(query.size(1)):
                step_mask, step_embs, trees, p_gen_mask, back_transform, index_list = self.get_step_biasing_embs_prefix(
                    targets[:, i], trees, lextrees)
                hptr, tcpgen_dist = self.get_meetingKB_emb_map(
                    query[:, i], step_mask, back_transform, index_list, meeting_KB=step_embs)
                hptrs.append(hptr)
                tcpgen_dists.append(tcpgen_dist)
                p_gen_masks.append(p_gen_mask)
            Hptr = torch.stack(hptrs, dim=1) # nutts * seq * hiddim
            tcpgen_dist = torch.stack(tcpgen_dists, dim=1)
            if self.useGPT:
                GPThid, GPTdist = GPThidden
                gen_prob = torch.softmax(self.pointer_gate(torch.cat([Hptr, hidden, GPThid], dim=-1)), dim=-1)
            else:
                gen_prob = torch.sigmoid(self.pointer_gate(torch.cat([Hptr, hidden], dim=-1)))
            p_gen_masks = torch.tensor(p_gen_masks).to(query.device).byte().t()

            model_dist = torch.softmax(logits[:, sotlen-1:-1], dim=-1)
            if self.useGPT:
                model_dist = model_dist * gen_prob[:, sotlen-1:-1, 0:1] + GPTdist * gen_prob[:, sotlen-1:-1, 1:2]
                model_dist = model_dist / gen_prob[:, sotlen-1:-1, 0:2].sum(dim=-1, keepdim=True)
                gen_prob = gen_prob[:, :, -1:]
            model_dist = model_dist.view(-1, model_dist.size(-1))
            loss, output = self.calc_ptr_loss(
                tcpgen_dist[:, sotlen-1:-1],
                model_dist,
                gen_prob[:, sotlen-1:-1],
                p_gen_masks[:, sotlen-1:-1],
                targets[:, sotlen:].reshape(-1),
            )
        else:
            output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)
            loss = F.nll_loss(output.view(-1, output.size(-1)), targets[:, sotlen:].reshape(-1))
        return loss, output


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    def getstates(self, mel: torch.Tensor, tokens: torch.Tensor)  -> Dict[str, torch.Tensor]:
        return self.decoder.get_states(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function


def lecun_normal_init_parameters(module):
    """Initialize parameters in the LeCun's manner."""
    for p in module.parameters():
        data = p.data
        if data.dim() == 1:
            # bias
            data.zero_()
        elif data.dim() == 2:
            # linear weight
            n = data.size(1)
            stdv = 1.0 / math.sqrt(n)
            data.normal_(0, stdv)
        elif data.dim() in (3, 4):
            # conv weight
            n = data.size(1)
            for k in data.size()[2:]:
                n *= k
            stdv = 1.0 / math.sqrt(n)
            data.normal_(0, stdv)
        else:
            raise NotImplementedError
