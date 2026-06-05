const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, LevelFormat, BorderStyle, WidthType,
  ShadingType, PageNumber, Header, Footer, PageBreak,
} = require('/opt/homebrew/lib/node_modules/docx');
const fs = require("fs");

const DARK_BLUE  = "1F3864";
const MID_BLUE   = "2E75B6";
const LIGHT_BLUE = "D5E8F0";
const PALE_BLUE  = "EBF3FB";
const GREY_BG    = "F2F2F2";
const WHITE      = "FFFFFF";
const BORDER_CLR = "BBBBBB";
const DUAL_BG    = "E8F4E8";  // light green for dual-stream callout

const thinBorder = { style: BorderStyle.SINGLE, size: 1, color: BORDER_CLR };
const allBorders = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };

const PAGE = { width: 12240, height: 15840 };
const CONTENT_W = 9360;

const NUMBERING = {
  config: [
    {
      reference: "bullet",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 }, spacing: { before: 40, after: 40 } } } }],
    },
    {
      reference: "sub-bullet",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 900, hanging: 270 }, spacing: { before: 30, after: 30 } } } }],
    },
    {
      reference: "number",
      levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 }, spacing: { before: 60, after: 60 } } } }],
    },
  ],
};

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, bold: true, size: 34, color: DARK_BLUE, font: "Arial" })],
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: MID_BLUE, space: 4 } } });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, bold: true, size: 28, color: MID_BLUE, font: "Arial" })] });
}
function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, size: 24, color: DARK_BLUE, font: "Arial" })] });
}
function body(runs_or_text, opts = {}) {
  const children = typeof runs_or_text === "string"
    ? [new TextRun({ text: runs_or_text, size: 22, font: "Arial" })]
    : runs_or_text;
  return new Paragraph({ children, spacing: { before: 60, after: 80 }, ...opts });
}
function run(text, { bold=false, color="000000", size=22, italic=false, font="Arial" }={}) {
  return new TextRun({ text, bold, color, size, italic, font });
}
function code(text) {
  return new TextRun({ text, font: "Courier New", size: 20, color: "8B0000" });
}
function bullet(runs_or_text) {
  const children = typeof runs_or_text === "string"
    ? [new TextRun({ text: runs_or_text, size: 22, font: "Arial" })]
    : runs_or_text;
  return new Paragraph({ numbering: { reference: "bullet", level: 0 }, children, spacing: { before: 40, after: 40 } });
}
function subBullet(text) {
  return new Paragraph({ numbering: { reference: "sub-bullet", level: 0 },
    children: [new TextRun({ text, size: 22, font: "Arial" })], spacing: { before: 30, after: 30 } });
}
function numbered(runs_or_text) {
  const children = typeof runs_or_text === "string"
    ? [new TextRun({ text: runs_or_text, size: 22, font: "Arial" })]
    : runs_or_text;
  return new Paragraph({ numbering: { reference: "number", level: 0 }, children, spacing: { before: 60, after: 60 } });
}
function spacer(n=1) {
  return Array.from({ length: n }, () =>
    new Paragraph({ children: [new TextRun("")], spacing: { before: 0, after: 60 } }));
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}
function callout(title, bodyText, bgColor = PALE_BLUE) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA }, borders: allBorders,
      shading: { fill: bgColor, type: ShadingType.CLEAR },
      margins: { top: 120, bottom: 120, left: 160, right: 160 },
      children: [
        new Paragraph({ children: [new TextRun({ text: title, bold: true, size: 22, font: "Arial", color: DARK_BLUE })], spacing: { before: 0, after: 60 } }),
        new Paragraph({ children: [new TextRun({ text: bodyText, size: 22, font: "Arial" })], spacing: { before: 0, after: 0 } }),
      ],
    })]})],
  });
}
function kvTable(rows, colW=[2800, 6560]) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: colW,
    rows: rows.map(([k, v], i) => new TableRow({ children: [
      new TableCell({ width: { size: colW[0], type: WidthType.DXA }, borders: allBorders,
        shading: { fill: i%2===0 ? LIGHT_BLUE : WHITE, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: k, bold: true, size: 21, font: "Arial", color: DARK_BLUE })] })] }),
      new TableCell({ width: { size: colW[1], type: WidthType.DXA }, borders: allBorders,
        shading: { fill: i%2===0 ? WHITE : GREY_BG, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: v, size: 21, font: "Arial" })] })] }),
    ]})),
  });
}
function qaBlock(q, a) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [
      new TableRow({ children: [new TableCell({ width: { size: CONTENT_W, type: WidthType.DXA }, borders: allBorders,
        shading: { fill: LIGHT_BLUE, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 60, left: 160, right: 160 },
        children: [new Paragraph({ children: [new TextRun({ text: "Q: "+q, bold: true, size: 22, font: "Arial", color: DARK_BLUE })], spacing: { before:0, after:0 } })] })] }),
      new TableRow({ children: [new TableCell({ width: { size: CONTENT_W, type: WidthType.DXA }, borders: allBorders,
        shading: { fill: WHITE, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 160, right: 160 },
        children: [new Paragraph({ children: [new TextRun({ text: "A: "+a, size: 22, font: "Arial" })], spacing: { before:0, after:0 } })] })] }),
    ],
  });
}

// ── Two-column phase label table (for block stack overview) ────────
function phaseRow(phase, label, desc, bgColor) {
  const colW = [1200, 2400, 5760];
  return new TableRow({ children: [
    new TableCell({ width: { size: colW[0], type: WidthType.DXA }, borders: allBorders,
      shading: { fill: bgColor, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: phase, bold: true, size: 21, font: "Arial", color: DARK_BLUE })] })] }),
    new TableCell({ width: { size: colW[1], type: WidthType.DXA }, borders: allBorders,
      shading: { fill: bgColor, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: label, bold: true, size: 21, font: "Arial" })] })] }),
    new TableCell({ width: { size: colW[2], type: WidthType.DXA }, borders: allBorders,
      shading: { fill: WHITE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: desc, size: 21, font: "Arial" })] })] }),
  ]});
}

// ═══════════════════════════════════════════════════════════════════
const doc = new Document({
  numbering: NUMBERING,
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, font: "Arial", color: DARK_BLUE },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: MID_BLUE },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: DARK_BLUE },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE.width, height: PAGE.height },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      children: [
        new TextRun({ text: "Boomer FLA — Architecture & Training Reference", size: 18, font: "Arial", color: "888888" }),
        new TextRun({ text: "\t", size: 18 }),
        new TextRun({ text: "Interview Preparation", size: 18, font: "Arial", color: "888888", italics: true }),
      ],
      tabStops: [{ type: "right", position: 9360 }],
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: MID_BLUE, space: 2 } },
    })]}) },
    footers: { default: new Footer({ children: [new Paragraph({
      children: [
        new TextRun({ text: "Confidential — Personal Use Only", size: 16, font: "Arial", color: "888888" }),
        new TextRun({ text: "\t", size: 16 }),
        new TextRun({ text: "Page ", size: 16, font: "Arial", color: "888888" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Arial", color: "888888" }),
      ],
      tabStops: [{ type: "right", position: 9360 }],
      border: { top: { style: BorderStyle.SINGLE, size: 4, color: MID_BLUE, space: 2 } },
    })]}) },
    children: [

      // ═══════════════════════════════════════════════════════════════
      // TITLE PAGE
      // ═══════════════════════════════════════════════════════════════
      ...spacer(3),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "Boomer FLA", bold: true, size: 72, font: "Arial", color: DARK_BLUE })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240 },
        children: [new TextRun({ text: "Architecture & Training — Deep Dive", size: 36, font: "Arial", color: MID_BLUE })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "Interview Preparation Reference", size: 26, font: "Arial", color: "555555", italics: true })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 0 },
        border: { top: { style: BorderStyle.SINGLE, size: 6, color: MID_BLUE, space: 4 } },
        children: [new TextRun({ text: "657M parameters  |  1024×1024px  |  bfloat16  |  GatedDeltaNet DiT + Dual-Stream Blocks", size: 22, font: "Arial", color: "444444" })] }),
      ...spacer(2),
      pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 1. SYSTEM OVERVIEW
      // ═══════════════════════════════════════════════════════════════
      h1("1  System Overview"),
      body([run("Boomer FLA is a text-to-image "), run("flow-matching", { bold: true }),
        run(" generative model that produces 1024×1024 px images from natural-language prompts. It is sometimes called a diffusion-style model, but training uses "),
        run("flow matching (velocity prediction in latent space)", { bold: true }),
        run(", not DDPM noise prediction. Three independent components work in sequence:")]),
      ...spacer(1),
      kvTable([
        ["Component",    "Role in the pipeline"],
        ["Gemma 4 2B",   "Text encoder — converts the prompt into 300 context-rich embeddings (1536-dim each)"],
        ["BoomerFLADiT", "Flow-matching transformer (DiT) — predicts the velocity field at each timestep; operates in compressed latent space"],
        ["DC-AE f32c32", "Variational autoencoder — encodes pixel images to 32×32 latents and decodes them back to 1024×1024 pixels"],
      ], [2200, 7160]),
      ...spacer(1),
      body([run("At inference: "), run("prompt → Gemma 4 2B → text embeddings → BoomerFLADiT (32 STORK-2 denoising steps) → latent → DC-AE decode → pixel image", { italic: true, color: "444444" }),
        run(".  Training targets only the DiT; VAE and text encoder are frozen throughout.")]),
      ...spacer(1),
      callout("Key insight for interviews",
        "The model never operates on pixels directly. DC-AE compresses 1024×1024 RGB (3×1024×1024) to 32×32×32 latents — a 32× spatial compression. The DiT sees only these compact latents, making it memory-efficient and enabling long-sequence linear attention."),
      ...spacer(1), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 2. VAE
      // ═══════════════════════════════════════════════════════════════
      h1("2  Variational Autoencoder — DC-AE f32c32"),
      body("The DC-AE (Deeply Compressed Autoencoder) from MIT-HAN Lab is a state-of-the-art VAE used by SANA. f32c32 means 32× spatial compression with 32 latent channels."),
      h2("2.1  Compression Mathematics"),
      body([run("Input image: "), code("3 × 1024 × 1024"), run("  →  Latent space: "), code("32 × 32 × 32"),
        run("  (32 channels, 32×32 spatial grid). The token count depends on resolution:")]),
      ...spacer(1),
      kvTable([
        ["512px pre-train",  "16×16 latent grid  →  256 tokens"],
        ["1024px fine-tune", "32×32 latent grid  →  1024 tokens  (4× more than pre-train)"],
        ["KL-VAE at 1024px (for comparison)", "128×128 latent grid  →  16,384 tokens (16× more than DC-AE)"],
      ], [3400, 5960]),
      body("DC-AE’s aggressive 32× compression is what makes the full-resolution, patch_size=1 token sequence affordable for the DiT."),
      h2("2.2  Output Range and Decode Arithmetic"),
      body("DC-AE outputs raw latents in the range [−1, +1]. Decoding requires careful normalisation:"),
      ...spacer(1),
      kvTable([
        ["Wrong approach",   "latent.clamp(0, 1) — destroys all negative values; shadows become solid black"],
        ["Correct approach", "(latent.float() / 2.0 + 0.5).clamp(0, 1) — linearly maps [−1,+1] to [0,1] safely"],
      ], [2800, 6560]),
      ...spacer(1),
      body([run("Before decoding, the DiT output is de-normalised from training-standardized space and divided by the scaling factor (0.41407): "),
        code("latent_dec = (latent × std + mean) / 0.41407")]),
      h2("2.3  Why Freeze the VAE?"),
      bullet("Training a VAE jointly with a DiT is expensive and often unstable"),
      bullet("A pre-trained VAE gives a stable latent space — DiT gradients never disturb it"),
      bullet("Latent statistics (mean/std per channel) are pre-computed once and cached; no VAE forward pass during training"),
      ...spacer(1),
      qaBlock("Why use DC-AE instead of the standard Stable Diffusion KL-VAE?",
        "DC-AE achieves 32× compression vs 8× for KL-VAE. At 1024px that means 32×32 latents vs 128×128, giving 16× fewer tokens. Fewer tokens = faster attention, smaller VRAM footprint per step. The trade-off is that DC-AE is harder to train from scratch, but since it is pre-trained and frozen we inherit all its benefits without any instability."),
      ...spacer(1), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 3. TEXT ENCODER
      // ═══════════════════════════════════════════════════════════════
      h1("3  Text Encoder — Gemma 4 2B"),
      h2("3.1  Why a Language Model as a Text Encoder?"),
      body("Traditional CLIP text encoders (77 tokens, 768-dim) are too weak for detailed prompts. Boomer follows SANA and Lumina-T2X in using a large language model decoder:"),
      bullet([run("Context length: ", { bold: true }), run("300 tokens vs CLIP’s 77 — supports long, detailed prompts")]),
      bullet([run("Embedding dim: ", { bold: true }), run("1536 per token (Gemma 4 2B hidden dim) — rich semantic features")]),
      bullet([run("Causal decoder: ", { bold: true }), run("later tokens attend to all prior context, encoding long-range prompt structure")]),
      h2("3.2  How Encoding Works"),
      numbered("Tokenize the prompt to up to 300 tokens (padded to exactly 300)"),
      numbered("Run through the Gemma 4 2B decoder; extract the full hidden-state sequence — shape (1, 300, 1536)"),
      numbered("Select: first token (pos 0) + last 299 positions — this 300-vector sequence is the conditioning input"),
      numbered([run("Project "), run("once at the start of the DiT forward pass", { bold: true }),
        run(" via CaptionEmbedder MLP (1536 → 896). All blocks then reuse these projected text tokens — for cross-attention in single-stream blocks, or joint attention in dual-stream blocks. The projection is NOT recomputed per block.")]),
      h2("3.3  Classifier-Free Guidance Null Condition"),
      body([run("During CFG inference, a "), run("learned null embedding", { bold: true }), run(" stored as a learnable "),
        code("nn.Parameter"), run(" of shape (1, 300, 1536) inside "), code("CaptionEmbedder"),
        run(" is used as the unconditional condition. Trained via CFG dropout (p=0.1).")]),
      h2("3.4  VRAM Considerations"),
      bullet("Fresh-prompt mode (16+ GB GPU): encode + generate in one call, peak ~13.38 GB"),
      bullet("Condition-cache mode (8 GB GPU): encode offline, save embedding, load only DiT + VAE at generation time (~4.76 GB peak)"),
      ...spacer(1),
      qaBlock("How does the model handle different prompt lengths?",
        "The tokenizer pads all prompts to exactly 300 tokens. An attention mask (1 for real, 0 for padding) is passed to every cross-attention block. Padding positions receive −10000 in the attention bias, so after softmax they contribute ~0 weight. Image tokens attend only to real text tokens."),
      ...spacer(1), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 4. BOOMER FLA DiT ARCHITECTURE
      // ═══════════════════════════════════════════════════════════════
      h1("4  BoomerFLADiT — Transformer Architecture"),
      body("The heart of Boomer is a 657M parameter Diffusion Transformer operating on 1024 latent tokens. It combines a FLUX-style dual-stream opening with a GatedDeltaNet (Flash Linear Attention) main stack."),

      h2("4.1  Configuration Snapshot"),
      kvTable([
        ["Parameters",           "657 million"],
        ["Depth",                "24 transformer blocks total"],
        ["Hidden dim",           "896"],
        ["Attention heads",      "14  (head dim = 64)"],
        ["MLp ratio",            "2.5  (inner dim = 2240)"],
        ["Patch size",           "1 — one token per latent pixel (1024 tokens at 1024px, 256 at 512px)"],
        ["Latent channels",      "32  (DC-AE output channels)"],
        ["Latent size",          "32×32 grid at 1024px"],
        ["Dual-stream depth",    "2  — blocks 0–1 are FLUX-style dual-stream (BoomerFLADualStreamBlock)"],
        ["Single-stream depth",  "22  — blocks 2–23 are standard single-stream (BoomerFLABlock)"],
        ["Parallel block",       "Yes — single-stream branches run in parallel (not sequential) and are summed"],
        ["Primary mixer",        "GatedDeltaNet (bidirectional FLA, chunk mode, conv_size=4)"],
        ["Short conv",           "Yes — 4-tap local conv inside GatedDeltaNet on every single-stream block"],
        ["SDPA image attention", "Every 6th SINGLE-STREAM block only: layers 5, 11, 17, 23 (0-indexed). Not in dual-stream blocks"],
        ["Position encoding",    "No absolute pos embed; multimodal coord RoPE in dual-stream blocks + 2D RoPE in SDPA layers"],
        ["Text dim",             "1536  (Gemma 4 2B)"],
        ["Text seq len",         "300 tokens"],
        ["Dtype",                "bfloat16"],
      ]),
      ...spacer(1),
      callout("Training flag that matters: --no-abs-pos-embed",
        "Absolute positional embeddings are disabled. Spatial position is encoded exclusively through: (1) multimodal coordinate RoPE in dual-stream blocks 0–1, (2) 2D RoPE in the periodic full image SDPA layers, and (3) implicit spatial structure in GLUMBConv’s depthwise 3×3 convolution."),
      ...spacer(1), pageBreak(),

      h2("4.2  Patching: Latent Pixels to Tokens"),
      body([run("With patch_size=1, each latent pixel becomes exactly one token. The "),
        code("x_embedder"), run(" is a simple "), code("nn.Linear(32, 896)"),
        run(" that maps the 32 channel values at each spatial position to an 896-dim hidden vector."),
        run(" The resulting token sequence has shape (B, 1024, 896) for 1024×1024 images.")]),
      body("No absolute positional embedding is added. Position comes entirely from the RoPE components described in Section 4.1."),
      ...spacer(1), pageBreak(),

      // 4.3  Block stack
      h2("4.3  Block Stack: Two Phases"),
      body("The 24 blocks are split into two distinct phases:"),
      ...spacer(1),
      new Table({
        width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [1200, 2400, 5760],
        rows: [
          new TableRow({ tableHeader: true, children: ["Phase", "Blocks", "Description"].map((h, i) =>
            new TableCell({ width: { size: [1200,2400,5760][i], type: WidthType.DXA }, borders: allBorders,
              shading: { fill: DARK_BLUE, type: ShadingType.CLEAR }, margins: { top:80, bottom:80, left:120, right:120 },
              children: [new Paragraph({ children: [new TextRun({ text: h, bold:true, size:21, font:"Arial", color:"FFFFFF" })] })] })) }),
          phaseRow("1", "Blocks 0–1  (2 total)", "FLUX-style dual-stream: joint image+text SDPA only. No GatedDeltaNet, no every-6th SDPA. Both streams updated", DUAL_BG),
          phaseRow("2", "Blocks 2–23  (22 total)", "Single-stream: GatedDeltaNet + cross-attn + GLUMBConv every block; full SDPA every 6th. Only image tokens updated", PALE_BLUE),
        ],
      }),
      ...spacer(1),
      callout("Where the full SDPA layers actually live",
        "The periodic full image SDPA (every 6th block) applies ONLY inside single-stream BoomerFLABlock layers 5, 11, 17, 23. The two dual-stream blocks (0–1) use joint text+image SDPA as their core attention — they have no GatedDeltaNet mixer and are not part of the every-6th SDPA count."),
      ...spacer(1),

      h3("4.3.1  Phase 1 — Dual-Stream Blocks (BoomerFLADualStreamBlock)"),
      body([run("Blocks 0 and 1 are FLUX-inspired ", { bold: false }),
        run("dual-stream blocks", { bold: true }),
        run(". Unlike all later blocks, both image and text token streams are updated at the same time:")]),
      bullet([run("Separate adaLN-Zero: ", { bold: true }),
        run("image and text each have their own scale_shift_table and mod MLP, driven by the same timestep embedding")]),
      bullet([run("Joint self-attention: ", { bold: true }),
        run("Q/K/V are computed for both streams, concatenated as [text ∥ image], and SDPA is run over the combined sequence. Outputs are split back and fed to each stream’s out_proj")]),
      bullet([run("Multimodal coordinate RoPE: ", { bold: true }),
        run("tags each token with (type=0 for text / type=1 for image, row, col) so the joint attention can distinguish text from image positions. This is the primary spatial positional signal in these blocks")]),
      bullet([run("Image FFN: ", { bold: true }),
        run("GLUMBConv (convolutional, spatial-aware)")]),
      bullet([run("Text FFN: ", { bold: true }),
        run("standard TokenMLP (token-independent)")]),
      bullet([run("Parallel block: ", { bold: true }),
        run("dual-stream blocks also honour --parallel-block — attention and MLP branches run in parallel from the same normalized base, just like single-stream blocks")]),
      body([run("Crucially, text tokens are ", { bold: false }),
        run("refined", { bold: true }),
        run(" in blocks 0–1 — the model can modify its understanding of the text given the noisy image context. After block 1, text becomes a static conditioning input accessed via cross-attention only.")]),
      ...spacer(1),

      h3("4.3.2  Phase 2 — Single-Stream Blocks (BoomerFLABlock) with Parallel Execution"),
      body([run("Blocks 2–23 follow a single-stream design where image tokens are the main stream. With "),
        run("--parallel-block", { bold: true }),
        run(", all four branches are computed simultaneously from the same normalized base and summed — not applied sequentially:")]),
      ...spacer(1),
      kvTable([
        ["Branch 1 (every block)",     "GatedDeltaNet bidirectional FLA self-mixer (linear attention, O(N) memory)"],
        ["Branch 2 (every 6th block)", "Full SDPA self-attention over all image tokens with 2D RoPE (quadratic but infrequent)"],
        ["Branch 3 (every block)",     "Sana-style cross-attention: image tokens attend to text embeddings (with QK RMSNorm)"],
        ["Branch 4 (every block)",     "GLUMBConv FFN: tokens reshaped to 32×32 grid, depthwise 3×3 conv + GLU, projected back"],
      ], [3200, 6160]),
      ...spacer(1),
      body([run("Parallel execution (DiT-parallel style): each branch sees "),
        code("modulate(norm(base), shift, scale)"),
        run(" independently, and the sum "),
        code("x = base + gate1*branch1 + gate2*branch2 + ..."),
        run(" replaces the sequential residual chain. This allows all branches to be computed simultaneously, improving throughput.")]),
      ...spacer(1),
      pageBreak(),

      // 4.4 GatedDeltaNet
      h2("4.4  GatedDeltaNet — Flash Linear Attention Mixer"),
      body("GatedDeltaNet replaces quadratic softmax self-attention with a sub-quadratic recurrence in all 22 single-stream blocks, keeping memory constant in sequence length."),
      h3("4.4.1  From Softmax to Linear Attention (background only)"),
      callout("Background context — Boomer does not use plain linear attention in production",
        "The mixer-type flag is fla_gated_deltanet, NOT fla_linear. The simple ReLU linear-attention formula below is shown only to build intuition for how sub-quadratic attention works. The actual production mixer is GatedDeltaNet, described in Section 4.4.2.", "FFF4E0"),
      ...spacer(1),
      body([run("Standard SDPA: "), code("Attention(Q,K,V) = softmax(QKᵀ / √d) V"), run(" — O(N²) memory.")]),
      body([run("Plain linear attention (illustrative): "), code("LinearAttn(Q,K,V) = φ(Q)(φ(K)ᵀV)"), run(" where φ = ReLU. Compute φ(K)ᵀV first (D×D matrix), reducing to O(N·D). Memory is linear in N. GatedDeltaNet builds on this idea but adds the delta rule, gating, and short conv described next.")]),
      h3("4.4.2  GatedDeltaNet Specifics"),
      bullet([run("Delta update rule: ", { bold: true }), run("state matrix S updated via content-weighted Delta rule; selectively overwrites old information")]),
      bullet([run("Gating: ", { bold: true }), run("sigmoid gate controls how much prior memory to retain — adaptive forgetting")]),
      bullet([run("Chunk mode: ", { bold: true }), run("sequence processed in hardware-parallel chunks, not token-by-token recurrence")]),
      bullet([run("Short conv (conv_size=4): ", { bold: true }), run("a 4-tap local depthwise convolution is applied to input tokens before the delta-net update, adding cheap local context to each position before the recurrence processes it")]),
      h3("4.4.3  Bidirectional Processing"),
      numbered("Forward pass: process tokens left-to-right through mixer_fwd"),
      numbered("Backward pass: flip the sequence, process right-to-left through mixer_bwd, flip back"),
      numbered("Concatenate both outputs along channels (896×2 = 1792)"),
      numbered("Project back to 896 via zero-initialized out_proj"),
      body("Zero initialization means the bidirectional branch starts as an identity — it gradually learns to contribute without destabilizing early training."),
      ...spacer(1),
      qaBlock("Why use linear attention (GatedDeltaNet) instead of SDPA for all layers?",
        "SDPA scales as O(N²). At N=1024 tokens with batch 24, attention maps alone take 24 × 14 heads × 1024² × 2 bytes ≈ 700 MB just for attention weights. GatedDeltaNet replaces this with a fixed D×D=64×64 memory matrix per head, collapsing VRAM to O(N). The short conv (conv_size=4) adds local context that the delta-net recurrence alone might miss."),
      ...spacer(1), pageBreak(),

      // 4.5 Full image SDPA
      h2("4.5  Periodic Full Image Self-Attention (Every 6th Block)"),
      body([run("GatedDeltaNet approximates — it cannot capture all global pairwise interactions. To compensate, blocks "),
        run("5, 11, 17, 23", { bold: true }), run(" (0-indexed, i.e., the 6th, 12th, 18th, 24th) insert full O(N²) SDPA over all 1024 image tokens:")]),
      bullet([run("2D RoPE: ", { bold: true }), run("head_dim splits in half — first half encodes row position, second half encodes column position via standard 1D RoPE applied to each axis separately")]),
      bullet([run("PyTorch SDPA kernel: ", { bold: true }), run("Flash Attention 2/3/4 optionally; falls back to scaled_dot_product_attention")]),
      bullet([run("Zero-initialized out_proj: ", { bold: true }), run("starts as a skip connection, gradually learns to contribute")]),
      ...spacer(1),
      qaBlock("Why every 6th block instead of every block?",
        "Full SDPA at N=1024: 1024² = 1M entries per head, ×14 heads = 14M per layer, ×24 layers = full quadratic budget. Spacing at every 6 blocks gives 4 global coherence checkpoints while keeping 20 of 24 layers as cheap O(N) linear attention. The fraction (4/24 ≈ 17% SDPA layers) is a practical memory vs quality trade-off."),
      ...spacer(1),

      // 4.6 adaLN
      h2("4.6  adaLN-Zero Timestep Conditioning"),
      body([run("Every block (both phases) uses adaptive LayerNorm-Zero (adaLN-Zero). A learnable "),
        code("scale_shift_table"), run(" (shape 9×896 in single-stream blocks, 6×896 each in dual-stream) stores per-block base offsets. The "),
        code("mod"), run(" MLP maps t_emb → dynamic adjustments. These sum and split into (shift, scale, gate) triplets for each branch:")]),
      body([code("modulate(x, shift, scale) = x × (1 + scale) + shift")]),
      body([run("Residuals: "), code("x = x + gate × branch_output"), run(". All "), code("mod"), run(" MLPs are zero-initialized — at step 0 every block is pure identity (pass-through), letting stable pre-training begin before conditioning gradually activates.")]),
      ...spacer(1),

      // 4.7 Cross-attention
      h2("4.7  Cross-Attention and Text Conditioning"),
      body([run("Present in all 22 single-stream blocks. Image tokens act as Q, projected text tokens as K and V. "),
        run("Sana-style design", { bold: true }), run(" ("), code("SanaMultiHeadCrossAttention"), run(") with QK RMSNorm:")]),
      h3("4.7.1  QK Norm"),
      body("Gemma 4 2B produces high-magnitude hidden states. Without clamping, Q·K dot products grow large — attention collapses to one token. AttentionRMSNorm is applied per-token to both Q and K after projection, capping magnitude while preserving direction."),
      h3("4.7.2  Zero-Initialized Out Projection"),
      body([run("Cross-attention "), code("out_proj"), run(" starts at zero. At initialization, cross-attention contributes "),
        run("exactly zero", { bold: true }), run(" to the residual. The DiT begins as a pure noise predictor with no text conditioning, then gradually learns to incorporate Gemma’s signals. Prevents early noisy Gemma gradients from corrupting FLA state matrices.")]),
      ...spacer(1),

      // 4.8 GLUMBConv
      h2("4.8  GLUMBConv — Spatial FFN"),
      body("Instead of a token-independent MLP, Boomer uses a convolutional FFN (borrowed from SANA) in the image stream:"),
      numbered([run("1×1 Conv expand: ", { bold: true }), run("channel expansion ×2.5, producing 2240×2 channels (pre-GLU split)")]),
      numbered([run("Reshape to 2D grid: ", { bold: true }), run("tokens laid out as 32×32 spatial map")]),
      numbered([run("Depthwise 3×3 Conv: ", { bold: true }), run("spatial mixing within each channel; sees the neighbourhood of each latent position")]),
      numbered([run("GLU gate: ", { bold: true }), run("split channels, output = half1 × SiLU(half2) — retains 2240 channels")]),
      numbered([run("1×1 Conv project: ", { bold: true }), run("back to 896 hidden dim; zero-initialized")]),
      body("Key advantage: the depthwise conv gives each token local neighbourhood awareness that a 1D MLP lacks."),
      ...spacer(1),
      pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 5. FLOW MATCHING
      // ═══════════════════════════════════════════════════════════════
      h1("5  Flow Matching — Training Objective"),
      body("Boomer is trained with flow matching: a modern alternative to DDPM that defines a straight-line path from noise to data."),
      h2("5.1  Core Idea"),
      body([run("Linear interpolation between clean image latent x₀ and pure noise ε:")]),
      body([code("x_t = (1 – t) × x₀ + t × ε   (t = σ ∈ [0,1])")], { alignment: AlignmentType.CENTER }),
      body("At t=0 we have a clean image; at t=1 we have pure noise. The target velocity (constant vector field):"),
      body([code("v_target = x₀ – ε")], { alignment: AlignmentType.CENTER }),
      body("The DiT predicts this velocity. At inference, integrating the predicted velocity from t=1 back to t=0 recovers the clean latent."),
      h2("5.2  Sigma Shift (flow_shift = 1.5)"),
      body([run("Shifts sampling distribution toward higher noise levels (t closer to 1), giving more training signal where visual content emerges:")]),
      body([code("shifted_σ = 1.5 × σ / (1 + (1.5 – 1) × σ)")], { alignment: AlignmentType.CENTER }),
      body("Both pre-training (JourneyDB, 512px) and fine-tuning (FineT2I, 1024px) used flow_shift = 1.5."),
      h2("5.3  Timestep Sampler — Plateau Logit-Normal"),
      bullet([run("Logit-normal mode: ", { bold: true }), run("sigmas concentrated around midpoint (t≈0.5), where the model does most perceptual work")]),
      bullet([run("Plateau: ", { bold: true }), run("flat plateau above the mode, ensuring high-noise regime receives consistent gradient signal")]),
      bullet([run("Parameters: ", { bold: true }), run("μ=0, σ=1")]),
      h2("5.4  Min-SNR Loss Weighting (γ = 5)"),
      body([run("Clips per-sample loss weight at γ=5 to prevent low-noise (high-SNR) samples dominating the gradient:")]),
      body([code("weight = min(SNR, 5) / SNR    SNR = (1–σ)² / σ²")], { alignment: AlignmentType.CENTER }),
      h2("5.5  CFG Dropout (p = 0.1)"),
      body([run("With p=0.1, the text embedding is replaced by the learned "), code("null_text_embedding"), run(" inside "),
        code("CaptionEmbedder"), run(". Teaches the model conditional vs unconditional generation, enabling CFG at inference.")]),
      ...spacer(1),
      qaBlock("How does flow matching differ from DDPM?",
        "DDPM uses a score function target (predicting added noise) and a Markov chain with hundreds of steps. Flow matching predicts a straight-line velocity field, allowing far fewer steps (32 vs 1000) and simpler training. The interpolation is deterministic and linear; DDPM adds noise multiplicatively with a cosine/linear schedule."),
      ...spacer(1), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 6. TRAINING PROCESS
      // ═══════════════════════════════════════════════════════════════
      h1("6  Training Process"),
      body([run("Training is "), run("two completely separate optimization runs", { bold: true }),
        run(": a from-scratch pre-train on 512px data (scheduled 100k, stopped at step 52,000), then a resolution-transfer fine-tune on 1024px data (schedule raised to 75k, stopped at step 55,000) initialized from that pre-train checkpoint. The final fine-tune checkpoint is the published model. The architecture (896 hidden / 24 depth / 14 heads BoomerFLADiT) is "),
        run("identical", { bold: true }), run(" in both stages — only the data geometry and training hyperparameters change.")]),
      ...spacer(1),

      h2("6.1  Stage 1 — JourneyDB Pre-train (512px)"),
      kvTable([
        ["Dataset",        "JourneyDB (∼3.8M images, 512px)"],
        ["Latents",        "16×16×32  →  256 tokens per image"],
        ["Steps",          "Scheduled for 100,000 — STOPPED at 52,000 (checkpoint latent_dit_step_52000.pt used for fine-tune)"],
        ["Batch size",     "96"],
        ["Learning rate",  "1.2e-4  (peak)"],
        ["LR schedule",    "Linear warmup 1000 steps → cosine decay over the 100k schedule (min_lr_ratio 0.1); stopped early at 52k"],
        ["Init",           "From scratch (random initialization)"],
        ["Rolling shards", "5 WebDataset tars cached locally at a time"],
        ["Run name",       "boomer_fla_journeydb_bidir_conv"],
        ["Purpose",        "Learn fundamental image structure, composition, text-visual alignment"],
      ]),
      ...spacer(1),

      h2("6.2  Stage 2 — FineT2I Fine-tune (1024px)"),
      kvTable([
        ["Dataset",        "FineT2I (∼600k curated images, 1024px)"],
        ["Latents",        "32×32×32  →  1024 tokens per image (4× more than pre-train)"],
        ["Steps",          "Schedule raised to 75,000 — STOPPED at 55,000; this final checkpoint is the PUBLISHED model"],
        ["Batch size",     "24  (4× fewer than pre-train because tokens are 4× — keeps activation memory constant)"],
        ["Learning rate",  "1e-4  (peak), fresh 1000-step warmup"],
        ["LR schedule",    "Cosine decay 1e-4 → 1e-5 over the 75k schedule (min_lr_ratio 0.1); stopped early at 55k"],
        ["Init",           "--resume latent_dit_step_52000.pt --resume-weights-only (from the 512px pre-train)"],
        ["Latent stats",   "New finetune_1024_latent_stats.json (re-computed for FineT2I)"],
        ["Rolling shards", "10 WebDataset tars cached locally at a time"],
        ["Run name",       "boomer_fla_finetune_1024_run1"],
        ["Purpose",        "High-resolution generation, fine detail, stronger prompt following"],
      ]),
      ...spacer(1),
      callout("Resolution transfer uses --resume-weights-only, not full resume",
        "The fine-tune is a NEW optimization run on new resolution + data, not a continuation of the pre-train schedule (100k scheduled, stopped at 52k). --resume-weights-only loads all compatible weights (FLA, FFN, norms, cross-attention K/V — text_dim=1536 is unchanged since the same Gemma 4 2B encoder is used) but RESETS the optimizer state, step counter, EMA, and LR schedule to zero. Because patch_size=1 and latent channels=32 are identical at both resolutions, the patch-embed Linear loads cleanly with zero reinit; only the grid-dependent RoPE tables re-initialize for the 32×32 grid."),
      ...spacer(1),
      body([run("Both stages use "), run("patch_size=1", { bold: true }), run(" — one token per latent pixel. This differs from models like DiT-XL (patch_size=2) because Boomer’s linear attention makes the full 1024-token sequence tractable at reasonable memory cost.")]),
      ...spacer(1), pageBreak(),

      h2("6.3  Shared Optimiser & Schedule Settings"),
      body("These hyperparameters are identical across both stages (only LR peak and batch size differ, per the tables above):"),
      kvTable([
        ["Optimiser",       "Fused AdamW (betas 0.9, 0.999)"],
        ["LR schedule type","linear-warmup-cosine, warmup 1000 steps"],
        ["Min LR ratio",    "0.1 — cosine decays to 10% of peak, NOT to zero"],
        ["Gradient clip",   "max norm = 0.3"],
        ["Precision",       "bfloat16 (mixed precision; gradients in float32)"],
        ["Flow shift",      "1.5"],
        ["Timestep sampler","plateau-logit-normal (μ=0, σ=1)"],
        ["Min-SNR γ",       "5.0"],
        ["CFG dropout",     "0.1"],
        ["EMA decay",       "0.999"],
        ["EMA update freq", "Every 8 steps (--ema-update-every 8), NOT every step"],
        ["Compile",         "torch.compile, mode max-autotune-no-cudagraphs"],
        ["Seed",            "42"],
        ["Hardware",        "RTX Pro 6000 Blackwell Server Edition (sm_120, 96 GB GDDR7)"],
      ]),
      ...spacer(1),
      callout("Why max-autotune-no-cudagraphs?",
        "CUDA graph capture is not safe on Blackwell (sm_120) with the current PyTorch build, so CUDA graphs are disabled. torch.compile kernel autotuning still applies for the speedup.", "FFF4E0"),
      ...spacer(1),

      h2("6.4  Memory Headroom & Gradient Checkpointing (Available, Not Used)"),
      body([run("Neither run used "), run("--grad-checkpoint", { bold: true }),
        run(". Stage 1 ran at batch 96 and Stage 2 at batch 24 — both fit comfortably in the 96 GB GPU without checkpointing.")]),
      body([run("Gradient checkpointing is available as an option for scaling batch size further. GatedDeltaNet’s chunk-mode backward pass stores full q/k/v per layer (∼1.4 GB/sample) without checkpointing, capping batch at ∼64; with checkpointing activations drop to ∼0.14 GB/sample, which would allow batches as large as 384 (∼61 GB). It was simply not needed at the batch sizes actually used.")]),
      ...spacer(1),

      h2("6.5  Exponential Moving Average (EMA, decay = 0.999)"),
      body([run("Shadow copy updated every 8 steps as: "), code("EMA ← 0.999 × EMA + 0.001 × param"),
        run(". Inference uses EMA weights, not live training weights. Stored in float32, converted to bfloat16 for HuggingFace distribution (~1.25 GB).")]),

      h2("6.6  Latent Normalization"),
      body([run("Each latent channel standardized to zero-mean, unit-variance using pre-computed per-channel statistics: "),
        code("normalized = (latent – mean) / std"), run(". Each stage uses its own stats file (512px vs 1024px FineT2I). Inverted before VAE decoding at inference.")]),
      ...spacer(1),
      qaBlock("Why does the fine-tune use a slightly lower LR (1e-4) than the pre-train (1.2e-4)?",
        "The fine-tune starts from converged pre-train weights, so a smaller step size avoids destroying learned features. But the fresh 1000-step warmup plus the large resolution shift (256 → 1024 tokens) means the model still needs a meaningful LR to adapt the grid-dependent components — hence near-original 1e-4 rather than something tiny. Both stages decay to 10% of peak via min_lr_ratio=0.1 (pre-train 1.2e-4→1.2e-5, fine-tune 1e-4→1e-5)."),
      ...spacer(1),
      qaBlock("Why gradient clipping at 0.3?",
        "GatedDeltaNet maintains a D×D=64×64 state matrix per head, updated every step. Without clipping, a single large gradient can corrupt the state matrix, spiking the loss and requiring many steps to recover. Clipping at 0.3 keeps state updates bounded. The aggressive clip (vs 1.0 standard in DiT training) reflects the sensitivity of recurrent state matrices."),
      ...spacer(1),
      qaBlock("What is the difference between --resume and --resume-weights-only?",
        "--resume restores everything: weights, optimizer state, step counter, EMA, and LR schedule position — used to continue an interrupted run. --resume-weights-only loads only the model weights and resets the rest to zero — used to START a new run (different resolution, data, or hyperparameters) from a good initialization. Boomer uses --resume-weights-only for the 512→1024 transfer, and plain --resume for resuming an interrupted fine-tune from its own checkpoint."),
      ...spacer(1), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 7. STORK-2 SAMPLER
      // ═══════════════════════════════════════════════════════════════
      h1("7  STORK-2 — Inference Sampler"),
      body("At inference, Boomer uses STORK-2, a high-order Runge-Kutta flow matching solver, instead of a simple Euler integrator."),
      h2("7.1  Why Not Just Use Euler?"),
      body("Euler integration accumulates error at each step. Higher-order methods like Runge-Kutta use multiple function evaluations per step to achieve lower error at the same step count."),
      h2("7.2  STORK-2 Configuration"),
      kvTable([
        ["solver_order",    "2  (second-order Runge-Kutta)"],
        ["derivative_order","1  (first derivative estimates)"],
        ["substeps (s)",    "5  (RK micro-steps per denoising step)"],
        ["prediction_type", "flow_prediction  (velocity field, not score/noise)"],
        ["Total NFE",       "32 × 5 = 160 model evaluations"],
        ["Recommended steps","32  (empirically best quality vs speed)"],
      ]),
      h2("7.3  Classifier-Free Guidance at Inference"),
      body("At each step the DiT is called twice — once with text embeddings (conditional) and once with the null embedding (unconditional):"),
      body([code("v_guided = v_uncond + λ × (v_cond – v_uncond)   λ = 4.5")], { alignment: AlignmentType.CENTER }),
      body([run("CFG rescale (0.5) prevents over-saturation at high λ by normalising the guided velocity std to match the conditional velocity: "),
        code("v_final = 0.5 × v_guided × (σ(v_cond) / σ(v_guided)) + 0.5 × v_guided")]),
      ...spacer(1),
      qaBlock("What is the CFG scale trade-off and how does CFG rescale help?",
        "Higher CFG scale (λ=5+) improves prompt following but causes over-saturation: the guided velocity has larger magnitude than any single conditioning case. CFG rescale (from Imagen paper) corrects this by normalizing the guided velocity to the same std as the pure conditional velocity, recovering accurate exposure while preserving improved text alignment."),
      ...spacer(1), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 8. MEMORY PROFILE
      // ═══════════════════════════════════════════════════════════════
      h1("8  Memory Profile"),
      body("Measured at 1024×1024px, bfloat16, STORK-2, CFG batch=2."),
      ...spacer(1),
      kvTable([
        ["DiT weights (EMA, bf16)",   "1.25 GB"],
        ["Gemma 4 2B text encoder",   "8.62 GB"],
        ["Denoising peak (CFG on)",   "1.36 GB"],
        ["VAE decode peak",           "3.51 GB"],
      ]),
      ...spacer(1),
      kvTable([
        ["Mode",             "Peak VRAM"],
        ["Fresh-prompt (TE + DiT + VAE resident)", "13.38 GB"],
        ["Condition-cache (DiT + VAE only)",        "4.76 GB"],
      ], [5000, 4360]),
      ...spacer(1),
      body([run("System RAM: ", { bold: true }), run("Gemma 4 2B requires ∼9 GB RAM during loading. Recommendation: 16 GB+ system RAM.")]),
      ...spacer(1), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 9. INTERVIEW Q&A
      // ═══════════════════════════════════════════════════════════════
      h1("9  Interview Q&A — Likely Questions"),
      ...spacer(1),
      qaBlock("Walk me through a single denoising step at inference.",
        "We start with noisy latent x_t ∈ R^{32×32×32}. The current sigma is read from the STORK schedule. Text is projected ONCE via CaptionEmbedder (1536→896) before the block loop. We batch: [x_t, x_t] with conditioning [null_emb, text_emb]. Blocks 0–1 (dual-stream) run joint image+text SDPA with coord RoPE — both streams updated. After block 1 text is no longer updated. Blocks 2–23 only refine image tokens via GatedDeltaNet + cross-attn (to the fixed text) + GLUMBConv + periodic full SDPA, all in parallel (parallel-block), producing velocity predictions [v_uncond, v_cond]. We compute v = v_uncond + 4.5×(v_cond-v_uncond), apply CFG rescale, then STORK integrates via RK sub-steps to get x_{t-Δt}. Repeat 32 times."),
      ...spacer(1),
      qaBlock("What is the purpose of the dual-stream blocks and how do they differ from the main blocks?",
        "The two dual-stream blocks (0–1) allow image and text tokens to interact bidirectionally — the model can refine its text representation given the noisy image, and simultaneously update image tokens with full text context. This is FLUX-style early fusion. After block 1, text becomes a static key/value source accessed only via cross-attention; it is no longer updated. The single-stream blocks then refine image tokens using the fixed text as guidance."),
      ...spacer(1),
      qaBlock("How does the model condition on the noise level (timestep)?",
        "The sigma is fed through TimestepEmbedder (Linear → SiLU → Linear) producing a 896-dim vector. This enters each block’s mod MLP, which outputs (shift, scale, gate) triplets for adaLN-Zero conditioning. All mod MLPs start zero-initialized — conditioning activates gradually as the model trains, preventing early instability."),
      ...spacer(1),
      qaBlock("What is adaLN-Zero and why is it preferred?",
        "Standard LayerNorm uses fixed learned affine params. adaLN computes them from the timestep embedding. The -Zero suffix means all conditioning MLPs initialize to zero: scale→0, gate→0. Initially every block is a pure pass-through, letting stable pre-training occur before conditioning gradually turns on. It avoids early instability from random conditioning parameters."),
      ...spacer(1),
      qaBlock("How is text information injected into image generation?",
        "First, the 300×1536 Gemma embeddings are projected ONCE to 300×896 by CaptionEmbedder at the start of the forward pass (not per block). These projected tokens are then reused everywhere: joint self-attention in the 2 dual-stream blocks (text + image concatenated for Q/K/V), and cross-attention in all 22 single-stream blocks (image=Q, text=K/V). The y-norm (RMSNorm on caption tokens, scale=0.01) and cross-attention QK norm prevent Gemma’s high-magnitude embeddings from destabilising attention. The CFG null embedding is stored at 1536-dim inside CaptionEmbedder and projected through the same y_proj."),
      ...spacer(1),
      qaBlock("What are the failure modes of this architecture?",
        "(1) Humans/portraits: faces and anatomy are inconsistent (training data was JourneyDB landscapes + FineT2I stock). (2) Animals, legible text, very small details: limited training coverage. (3) HDR/painterly bias in landscapes from post-processed stock imagery. (4) Resolution-fixed: no abs pos embed is used, but 2D RoPE in SDPA layers assumes a fixed 32×32 grid; arbitrary resolutions would require interpolation. (5) CFG scale above 5 without CFG rescale causes oversaturation."),
      ...spacer(1),
      qaBlock("Why bfloat16 instead of float16?",
        "bfloat16 has the same exponent range as float32 (8 bits) with 7 mantissa bits vs float16’s 5 exponent / 10 mantissa. This means bfloat16 rarely overflows/underflows during backward passes (gradient spikes are common in diffusion training). float16 would require careful loss scaling. Reduced mantissa precision is acceptable since gradients need direction, not high precision."),
      ...spacer(1),
      qaBlock("What is the difference between self-attention and cross-attention in this model?",
        "Self-attention (GatedDeltaNet / full SDPA) lets image tokens interact with each other — enabling spatial reasoning and content structure. Cross-attention lets image tokens query text tokens to pull in semantic information. In dual-stream blocks both happen jointly in a single operation. In single-stream blocks they are separate branches. Text tokens are never updated after block 1."),
      ...spacer(1), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // 10. DESIGN DECISIONS SUMMARY
      // ═══════════════════════════════════════════════════════════════
      h1("10  Key Design Decision Summary"),
      ...spacer(1),
      new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [2400, 3000, 3960],
        rows: [
          new TableRow({ tableHeader: true,
            children: ["Decision", "Choice", "Rationale"].map((h, i) =>
              new TableCell({ width: { size: [2400,3000,3960][i], type: WidthType.DXA }, borders: allBorders,
                shading: { fill: DARK_BLUE, type: ShadingType.CLEAR }, margins: { top:80, bottom:80, left:120, right:120 },
                children: [new Paragraph({ children: [new TextRun({ text: h, bold:true, size:21, font:"Arial", color:"FFFFFF" })] })] })) }),
          ...([
            ["Dual-stream blocks 0-1", "FLUX-style joint attn",         "Text is updated alongside image in early layers; richer early fusion than pure cross-attn."],
            ["Primary mixer",          "GatedDeltaNet (linear attn)",   "O(N) memory vs O(N²) SDPA. 1024 tokens at batch 24 is tractable."],
            ["Bidirectional FLA",      "Forward + backward pass",       "Image tokens have no causal order. Both scan directions give global spatial context."],
            ["Short conv (size=4)",    "Local 4-tap depthwise conv",    "Adds cheap local context to each token before the delta-net recurrence."],
            ["Parallel block",         "Branches summed, not sequential","All branches see the same normalized input; faster throughput, DiT-parallel style."],
            ["SDPA every 6th",         "Full attn at layers 5,11,17,23","Global coherence checkpoints without paying full SDPA cost every layer."],
            ["No abs pos embed",       "RoPE only",                     "Coord RoPE in dual-stream + 2D RoPE in SDPA layers provide position; abs embed would conflict."],
            ["GLUMBConv FFN",          "Convolutional spatial FFN",     "Depthwise 3×3 encodes local spatial structure; standard MLP operates token-independently."],
            ["Patch size = 1",         "No patch grouping",             "Maximum spatial resolution in latent space; linear attn makes 1024 tokens affordable."],
            ["DC-AE f32c32",           "32× VAE compression",          "32×32 latents vs 128×128 for KL-VAE; 16× fewer tokens."],
            ["Gemma 4 2B TE",          "300 tokens, 1536-dim",          "Captures long complex prompts that CLIP (77 tok) cannot."],
            ["Min-SNR γ=5",            "Loss weight clipping",          "Balances gradient contribution; prevents high-SNR dominance."],
            ["Plateau logit-normal",   "Timestep sampler",              "Concentrates training around midrange sigmas where perceptual detail emerges."],
            ["STORK-2, 32 steps",      "High-order RK sampler",         "Lower error per step vs Euler; fewer steps for equivalent quality."],
            ["CFG rescale 0.5",        "Velocity std normalization",    "Prevents oversaturation at high CFG scale without reducing prompt adherence."],
          ]).map(([a, b, c], i) => new TableRow({ children: [a,b,c].map((text, j) =>
            new TableCell({ width: { size: [2400,3000,3960][j], type: WidthType.DXA }, borders: allBorders,
              shading: { fill: i%2===0 ? WHITE : GREY_BG, type: ShadingType.CLEAR }, margins: { top:80, bottom:80, left:120, right:120 },
              children: [new Paragraph({ children: [new TextRun({ text, size:20, font:"Arial" })] })] })) })),
        ],
      }),
      ...spacer(2), pageBreak(),

      // ═══════════════════════════════════════════════════════════════
      // APPENDIX
      // ═══════════════════════════════════════════════════════════════
      h1("Appendix — Key Formulas"),
      h2("A.1  Flow Matching Noisy Latent"),
      body([code("x_t = (1 – σ) × x₀ + σ × ε    ε ~ N(0,I),  σ = noise level")]),
      h2("A.2  Sigma Shift"),
      body([code("shifted_σ = 1.5 × σ / (1 + 0.5 × σ)")]),
      h2("A.3  adaLN Modulate"),
      body([code("modulate(x, shift, scale) = x × (1 + scale) + shift")]),
      h2("A.4  CFG Guidance"),
      body([code("v = v_uncond + 4.5 × (v_cond – v_uncond)")]),
      h2("A.5  CFG Rescale"),
      body([code("v_final = 0.5 × v × (σ(v_cond) / σ(v)) + 0.5 × v")]),
      h2("A.6  VAE Decode"),
      body([code("pixels = (vae.decode(latent_denorm).float() / 2.0 + 0.5).clamp(0,1)")]),
      h2("A.7  EMA Update"),
      body([code("EMA ← 0.999 × EMA + 0.001 × param")]),
      h2("A.8  Min-SNR Weight"),
      body([code("SNR = (1–σ)² / σ²    weight = min(SNR, 5) / SNR")]),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(
    "/Users/ak/Documents/genaiexperiments/ImageModel/Boomer/Boomer_Architecture_Interview_Guide.docx",
    buf
  );
  console.log("Done.");
});
