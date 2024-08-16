### ATTENTION

To extract attention through Transformer use the
`TransformerEncoderLayer`.

Important flags are:

- `need_weights` which should be true to fill the variables with weights
- `average_attn_weights` if True each head would expose its weights, otherwise all the heads in a layer will be averaged

To use this Class properly see the exaple:

```
encoder_layer = TransformerEncoderLayer(d_model=512, nhead=1, need_weights=True,
                                            average_attn_weights=False)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
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
print(transformer_model.encoder.layers[0].attention_weights.shape)  # 32,10,10
# or with average_attn_weights=False -> 32,8,10,10
   ```