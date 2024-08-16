from typing import Optional, Callable, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    r"""TransformerEncoderLayer modified to expose the attention weights.
    If used with default values emulates the torch one. Flags which make it act differently are:
    need_weights and average_attn_weights.

    TransformerEncoderLayer is made up of self-attn and feedforward network.
        This standard encoder layer is based on the paper "Attention Is All You Need".
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
        Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
        Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
        in a different way during application.

        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of the intermediate layer, can be a string
                ("relu" or "gelu") or a unary callable. Default: relu
            layer_norm_eps: the eps value in layer normalization components (default=1e-5).
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
            norm_first: if ``True``, layer norm is done prior to attention and feedforward
                operations, respectively. Otherwise it's done after. Default: ``False`` (after).
            need_weights: if ``True`` the layer save internally the attention weights for each head. Default: ``False``
            average_attn_weights: if ``True`` the layer will average all heads attention weights. Default: ``True``

        Examples::
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            >>> src = torch.rand(10, 32, 512)
            >>> out = encoder_layer(src)

        Alternatively, when ``batch_first`` is ``True``:
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            >>> src = torch.rand(32, 10, 512)
            >>> out = encoder_layer(src)

        Fast path:
            forward() will use a special optimized implementation described in
            `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
            conditions are met:

            - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
              argument ``requires_grad``
            - training is disabled (using ``.eval()``)
            - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
            - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
            - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
            - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
              nor ``src_key_padding_mask`` is passed
            - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
              unless the caller has manually modified one without modifying the other)

            If the optimized implementation is in use, a
            `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
            passed for ``src`` to represent padding more efficiently than using a padding
            mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
            returned, and an additional speedup proportional to the fraction of the input that
            is padding can be expected.

            .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
             https://arxiv.org/abs/2205.14135

        """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, need_weights: bool = False, average_attn_weights: bool = True,
                 ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first,
                         norm_first, device, dtype)
        self.attention_weights = None
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
                  ) -> Tensor:
        x, attn_weights = self.self_attn(x, x, x,
                                         attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask,
                                         need_weights=self.need_weights, average_attn_weights=self.average_attn_weights)
        self.attention_weights = attn_weights
        return self.dropout1(x)


class TransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    r"""TransformerDecoderLayer modified to expose the attention weights.
    If used with default values emulates the torch one. Flags which make it act differently are:
    need_weights and average_attn_weights.

    TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        need_weights: if ``True`` the layer save internally the attention weights for each head. Default: ``False``
        average_attn_weights: if ``True`` the layer will average all heads attention weights. Default: ``True``

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, need_weights: bool = False, average_attn_weights: bool = True, ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first,
                         norm_first, device, dtype)
        self.attention_weights = None
        self.multi_head_attention_weights = None
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False
                  ) -> Tensor:
        x, attn_weights = self.self_attn(x, x, x,
                                         attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask,
                                         is_causal=is_causal,
                                         need_weights=self.need_weights, average_attn_weights=self.average_attn_weights)
        self.attention_weights = attn_weights
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False
                   ) -> Tensor:
        x, attn_weights = self.multihead_attn(x, mem, mem,
                                              attn_mask=attn_mask,
                                              key_padding_mask=key_padding_mask,
                                              is_causal=is_causal,
                                              need_weights=self.need_weights,
                                              average_attn_weights=self.average_attn_weights)
        self.multi_head_attention_weights = attn_weights
        return self.dropout2(x)


if __name__=='__main__':
    encoder_layer = TransformerEncoderLayer(d_model=512, nhead=1, need_weights=True,
                                            average_attn_weights=False)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8, need_weights=True,
                                            average_attn_weights=False)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    transformer_model = nn.Transformer(nhead=2, num_encoder_layers=1, custom_encoder=transformer_encoder,
                                       custom_decoder=transformer_decoder)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((22, 32, 512))
    out = transformer_model(src, tgt)

    # transformer_model.encoder
    print(f'out shape')
    print(out.shape)
    print(f'ENCODER layer 0, self attention shape')
    print(transformer_model.encoder.layers[0].attention_weights.shape)  # 32, 1,10,10
    # or with average_attn_weights=False -> 32,8,10,10
    # 32 batch
    print(f'DECODER layer 0, self attention shape')
    print(transformer_model.decoder.layers[0].attention_weights.shape)
    print(f'DECODER layer 0, multi head attention shape')
    print(transformer_model.decoder.layers[0].multi_head_attention_weights.shape)