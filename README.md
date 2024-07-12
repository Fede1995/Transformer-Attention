### ATTENTION

To extract attention through Transformer use the 
`TransformerEncoderLayer_withAttention`. 

Important flags are:
- `need_weights` which should be true to fill the variables with weights
- `average_attn_weights` if True each head would expose its weights, otherwise all the heads in a layer will be averaged

To use this Class properly see the exaple:

`encoder_layer = TransformerEncoderLayer_withAttention(d_model=512, nhead=8, need_weights=True, average_attn_weights=False)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    transformer_model = nn.Transformer(nhead=2, num_encoder_layers=1, custom_encoder=transformer_encoder)`