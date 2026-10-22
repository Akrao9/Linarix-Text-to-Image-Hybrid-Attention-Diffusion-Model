# Architecture

Boomer V8 is a flow-matching text-to-image model. The trainable denoiser has
approximately 890M parameters; Qwen and DC-AE are frozen components.

| Component | V8 configuration |
| --- | --- |
| Denoiser | 28 blocks, width 1152, 18 heads, MLP ratio 2.5 |
| Condition encoder | 22 blocks; first 4 update both image and text streams |
| Linear mixer | Bidirectional GatedDeltaNet-2 with short convolutions |
| Joint refresh | Encoder blocks 7, 11, 15, 21; zero-based indices |
| Velocity decoder | 6 blocks with full image attention and GLUMBConv |
| Residual routing | Block AttnRes over encoder block contributions |
| Text | Qwen3.5-4B layers 9, 18, 27; normalized and concatenated to 7680 features |
| Text length | 320 tokens, including instruction and chat template |
| Image latent | DC-AE f32c32; 32 channels and 32× spatial compression |

## Forward pass

1. Embed the noisy latent and timestep. Project Qwen features into the model width.
2. Run the dual-stream prefix, then linear mixer and joint refresh blocks.
   Text cross-attention also conditions the ordinary encoder blocks.
3. Aggregate the encoder's block contributions with Block AttnRes to produce `z`.
4. Re-embed the noisy latent for the velocity decoder. Each decoder block receives
   token-wise modulation from `s = silu(z + timestep_embedding)`.
5. Predict the flow velocity in latent space.

The encoder uses Hilbert order inside the linear mixers. Attention and spatial
GLUMBConv operations retain raster order. Shared adaLN, Fourier timestep features,
sandwich RMSNorm, and multimodal coordinate RoPE are enabled in the V8 recipes.

The model is hybrid: the prefix, refresh blocks, and decoder still use full
attention. Its total attention cost is not linear in image-token count.

| Image size | Latent grid | Image tokens, patch size 1 | Recipe RoPE factor |
| --- | --- | --- | --- |
| 512px | 16 × 16 | 256 | 1.0 |
| 1024px | 32 × 32 | 1024 | 2.0 |
| 2048px | 64 × 64 | 4096 | 4.0 |

## Training and inference

Training interpolates clean latents toward Gaussian noise:
`x_t = (1 - t) * x_0 + t * noise`. The velocity target is `noise - x_0`.
The V8 recipes use CMuon with auxiliary AdamW and an EMA of the denoiser.

The training model reserves 33 extra input channels for a masked latent and mask.
Text-to-image training supplies zeros there. The text-to-image export removes
those columns and the optional REPA head.

Sampling supports classifier-free guidance, guidance intervals, CFG rescaling,
and STORK/Euler/Heun solvers. The CLI can optionally reuse DDT encoder states
between steps; the exported pipeline recomputes the full model at each step.

## Sources

- [Model](../boomer/models/boomer_fla_dit.py) and [Qwen readout](../boomer/models/qwen_connector.py)
- [Training](../train.py), [sampling](../boomer/sampling.py), and [export](../tools/export_to_hf.py)
- [GatedDeltaNet-2 / FLA](https://github.com/fla-org/flash-linear-attention),
  [DDT](https://arxiv.org/abs/2504.05741),
  [Block AttnRes](https://arxiv.org/abs/2603.15031),
  [STORK](https://github.com/ZT220501/STORK)
