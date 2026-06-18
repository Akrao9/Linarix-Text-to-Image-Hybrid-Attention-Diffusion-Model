const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";

const BG = "0A0A0A", CARD = "151515", CARD2 = "1C1C1C", LINE = "2E2E2E";
const TXT = "FFFFFF", BODY = "CBCBCB", MUT = "8E8E8E";
const LIME = "C2EF4B", ROSE = "E8736B";
const HF = "Arial Black", BF = "Arial", MF = "Consolas";
const W = 13.333, H = 7.5, M = 0.7;

function bg(s) { s.background = { color: BG }; }
function title(s, runs, y) {
  s.addText(runs.map(r => ({ text: r[0], options: { color: r[1] } })),
    { x: M, y: y || 0.55, w: 12.2, h: 0.85, fontFace: HF, fontSize: 31, bold: true });
}
function foot(s, n) {
  s.addText("BOOMER AI", { x: M, y: H - 0.48, w: 3, h: 0.3, fontFace: BF, fontSize: 9, color: MUT, charSpacing: 2 });
  s.addText(String(n).padStart(2, "0"), { x: W - 1.0, y: H - 0.48, w: 0.5, h: 0.3, fontFace: BF, fontSize: 10, color: LIME, align: "right", bold: true });
}
function card(s, x, y, w, h, fill) { s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: fill || CARD }, line: { color: LINE, width: 1 } }); }
function callout(s, y, h, runs) {
  s.addShape(p.ShapeType.roundRect, { x: M, y, w: W - 2 * M, h, rectRadius: 0.06, fill: { color: CARD }, line: { color: LIME, width: 1.5 } });
  s.addText(runs.map(r => ({ text: r[0], options: { color: r[1], bold: r[2] || false } })),
    { x: M + 0.35, y, w: W - 2 * M - 0.7, h, fontFace: BF, fontSize: 15, valign: "middle", lineSpacingMultiple: 1.1 });
}
function tableRows(s, x, y, colW, rows, opt) {
  opt = opt || {};
  let ty = y;
  rows.forEach((r, ri) => {
    const head = ri === 0, boom = opt.boomRow === ri;
    const rh = r._h || (head ? 0.55 : 0.82);
    const fill = head ? CARD2 : (boom ? "20300F" : (ri % 2 ? "111111" : CARD));
    s.addShape(p.ShapeType.rect, { x, y: ty, w: colW.reduce((a, b) => a + b, 0), h: rh, fill: { color: fill }, line: { color: LINE, width: 0.75 } });
    if (boom) s.addShape(p.ShapeType.rect, { x, y: ty, w: 0.06, h: rh, fill: { color: LIME } });
    let cx = x;
    r.cells.forEach((c, ci) => {
      s.addText(c, {
        x: cx + 0.2, y: ty, w: colW[ci] - 0.35, h: rh,
        fontFace: head ? HF : BF, fontSize: head ? 11.5 : 12.5,
        bold: head || (boom && ci === 0), color: head ? LIME : (boom ? TXT : (ci === 0 ? TXT : BODY)),
        valign: "middle", charSpacing: head ? 0.5 : 0, lineSpacingMultiple: 1.02,
      });
      cx += colW[ci];
    });
    ty += rh;
  });
  return ty;
}

// ===================================================== 1 TITLE
let s = p.addSlide(); bg(s);
s.addText("PRE-SEED  ·  CONFIDENTIAL", { x: M, y: 0.8, w: 8, h: 0.3, fontFace: BF, fontSize: 11, color: MUT, charSpacing: 2 });
s.addText("Boomer AI", { x: M, y: 2.3, w: 11, h: 1.3, fontFace: HF, fontSize: 64, bold: true, color: TXT });
s.addText("Low-Cost 1MP + 4MP Image Generation & Editing API", { x: M, y: 3.62, w: 12.2, h: 0.7, fontFace: HF, fontSize: 24, bold: true, color: LIME });
s.addText("Attention-light GDN2 diffusion for generation, editing, inpainting, and reference-image workflows.",
  { x: M, y: 4.45, w: 10.8, h: 0.6, fontFace: BF, fontSize: 15.5, color: BODY, lineSpacingMultiple: 1.15 });
s.addShape(p.ShapeType.line, { x: M, y: 5.45, w: 5.0, h: 0, line: { color: LINE, width: 1 } });
s.addText([
  { text: "~700M GDN2 prototype", options: { color: TXT, bold: true } }, { text: "    ·    ", options: { color: LINE } },
  { text: "1024px today, 4MP next", options: { color: TXT, bold: true } }, { text: "    ·    ", options: { color: LINE } },
  { text: "trained on a single GPU", options: { color: TXT, bold: true } },
], { x: M, y: 5.65, w: 12, h: 0.4, fontFace: BF, fontSize: 13 });

// ===================================================== 2 PROBLEM
s = p.addSlide(); bg(s);
title(s, [["High-Res Image AI Is ", TXT], ["Expensive to Serve", LIME]]);
s.addText("Quality keeps rising — but the high-resolution and reference workflows customers pay most for are the costliest to run.",
  { x: M, y: 1.55, w: 11.9, h: 0.5, fontFace: BF, fontSize: 16, color: MUT });
const probs = [
  ["High-res inflates cost", "Every extra megapixel multiplies latent tokens — dense attention makes 4MP serving VRAM-heavy and slow."],
  ["Reference & editing multiply it", "Conditioning on reference images balloons tokens per call, so the most valuable workflows cost the most to deliver."],
  ["Control is the real gap", "Enterprises need reliable editing, inpainting, and brand consistency — not just one-shot text-to-image."],
];
probs.forEach((d, i) => {
  const y = 2.5 + i * 1.32;
  s.addShape(p.ShapeType.roundRect, { x: M, y: y + 0.02, w: 0.5, h: 0.5, rectRadius: 0.06, fill: { color: CARD2 }, line: { color: ROSE, width: 1 } });
  s.addText("0" + (i + 1), { x: M, y: y + 0.02, w: 0.5, h: 0.5, fontFace: HF, fontSize: 13, color: ROSE, bold: true, align: "center", valign: "middle" });
  s.addText(d[0], { x: M + 0.8, y: y, w: 5.1, h: 0.55, fontFace: HF, fontSize: 18, bold: true, color: TXT, valign: "middle" });
  s.addText(d[1], { x: M + 6.1, y: y - 0.05, w: W - M - 6.1 - M + 0.7, h: 1.1, fontFace: BF, fontSize: 14, color: BODY, valign: "middle", lineSpacingMultiple: 1.12 });
});
foot(s, 2);

// ===================================================== 3 QUADRATIC vs LINEAR
s = p.addSlide(); bg(s);
title(s, [["Why Cost Explodes: ", TXT], ["Quadratic vs Linear", LIME]]);
s.addText("Softmax attention scales with the square of token count — and every megapixel adds tokens.",
  { x: M, y: 1.55, w: 11.9, h: 0.5, fontFace: BF, fontSize: 15.5, color: MUT });
const expl = [
  ["Softmax is O(N²)", "Every token attends to every other — memory and compute grow with the square of the token count.", ROSE],
  ["Resolution compounds it", "1MP → 4MP is ~4× the tokens, which is roughly 16× the attention cost. High-res is where it bites.", ROSE],
  ["Boomer stays near-linear", "GDN2 state mixing replaces the quadratic blow-up; exact attention runs only in a few bounded anchor blocks.", LIME],
];
expl.forEach((d, i) => {
  const y = 2.35 + i * 1.05;
  s.addShape(p.ShapeType.rect, { x: M, y: y + 0.04, w: 0.06, h: 0.8, fill: { color: d[2] } });
  s.addText(d[0], { x: M + 0.25, y: y, w: 5.2, h: 0.35, fontFace: HF, fontSize: 14.5, bold: true, color: d[2] === LIME ? LIME : TXT, valign: "top" });
  s.addText(d[1], { x: M + 0.25, y: y + 0.36, w: 5.25, h: 0.72, fontFace: BF, fontSize: 12, color: BODY, valign: "top", lineSpacingMultiple: 1.1 });
});
const cx0 = 7.0, cy0 = 5.5, cw = 5.3, chh = 2.55;
s.addShape(p.ShapeType.roundRect, { x: 6.5, y: 2.2, w: 6.0, h: 3.85, rectRadius: 0.06, fill: { color: CARD }, line: { color: LINE, width: 1 } });
s.addShape(p.ShapeType.line, { x: cx0, y: cy0, w: cw, h: 0, line: { color: LINE, width: 1.25 } });
s.addShape(p.ShapeType.line, { x: cx0, y: cy0 - chh, w: 0, h: chh, line: { color: LINE, width: 1.25 } });
s.addText("high", { x: cx0 - 0.62, y: cy0 - chh - 0.04, w: 0.52, h: 0.22, fontFace: BF, fontSize: 8, color: MUT, align: "right" });
s.addText("low", { x: cx0 - 0.62, y: cy0 - 0.2, w: 0.52, h: 0.22, fontFace: BF, fontSize: 8, color: MUT, align: "right" });
const xs = [0.04, 0.37, 0.69, 0.99].map(f => cx0 + f * cw);
const soft = [0.04, 0.20, 0.54, 0.96].map(f => cy0 - f * chh);
const boom = [0.02, 0.08, 0.17, 0.33].map(f => cy0 - f * chh);
function poly(pts, color) {
  for (let i = 0; i < pts.length - 1; i++) {
    s.addShape(p.ShapeType.line, { x: xs[i], y: pts[i + 1], w: xs[i + 1] - xs[i], h: pts[i] - pts[i + 1], line: { color, width: 3.25 }, flipV: true });
  }
  pts.forEach((yy, i) => s.addShape(p.ShapeType.ellipse, { x: xs[i] - 0.055, y: yy - 0.055, w: 0.11, h: 0.11, fill: { color }, line: { type: "none" } }));
}
poly(soft, ROSE);
poly(boom, LIME);
const xl = ["256px", "1MP", "4MP", "video"];
xs.forEach((x, i) => s.addText(xl[i], { x: x - 0.6, y: cy0 + 0.08, w: 1.2, h: 0.24, fontFace: BF, fontSize: 9.5, color: MUT, align: "center" }));
s.addShape(p.ShapeType.rect, { x: cx0 + 0.15, y: 2.45, w: 0.2, h: 0.2, fill: { color: ROSE }, line: { type: "none" } });
s.addText("Standard Softmax  O(N²)", { x: cx0 + 0.42, y: 2.41, w: 3.0, h: 0.28, fontFace: BF, fontSize: 10, color: BODY, valign: "middle" });
s.addShape(p.ShapeType.rect, { x: cx0 + 0.15, y: 2.78, w: 0.2, h: 0.2, fill: { color: LIME }, line: { type: "none" } });
s.addText("Boomer Hybrid  ~O(N)", { x: cx0 + 0.42, y: 2.74, w: 3.0, h: 0.28, fontFace: BF, fontSize: 10, color: BODY, valign: "middle" });
callout(s, 6.2, 0.72, [["The gap widens at every megapixel — ", LIME, true], ["exactly where Boomer's attention-light design compounds its cost advantage.", TXT]]);
foot(s, 3);

// ===================================================== 4 KEY ARCHITECTURE: WHY GDN2
s = p.addSlide(); bg(s);
title(s, [["Key Architecture: ", TXT], ["Why GDN2", LIME]]);
s.addText("Vanilla linear attention validates cheap high-res generation. GDN2 keeps that efficiency, adds learned memory control, and stays cheaper to serve as we scale up.",
  { x: M, y: 1.55, w: 11.9, h: 0.55, fontFace: BF, fontSize: 15, color: MUT, lineSpacingMultiple: 1.1 });
const cwid = (W - 2 * M - 0.3) / 2;
card(s, M, 2.2, cwid, 2.9);
s.addText("VANILLA LINEAR ATTENTION", { x: M + 0.3, y: 2.38, w: cwid - 0.6, h: 0.3, fontFace: HF, fontSize: 12, color: MUT, charSpacing: 0.5, bold: true });
s.addText(["Efficient token mixing", "Lower memory than softmax", "Strong for high-res generation"].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, color: BODY } })),
  { x: M + 0.3, y: 2.82, w: cwid - 0.6, h: 1.25, fontFace: BF, fontSize: 13.5, lineSpacingMultiple: 1.3, valign: "top" });
s.addText("Validated by Sana — linear attention + 32× DC-AE, up to 4096².", { x: M + 0.3, y: 4.45, w: cwid - 0.6, h: 0.5, fontFace: BF, fontSize: 11.5, italic: true, color: MUT, valign: "top", lineSpacingMultiple: 1.05 });
const rxc = M + cwid + 0.3;
card(s, rxc, 2.2, cwid, 2.9, CARD2);
s.addShape(p.ShapeType.rect, { x: rxc, y: 2.2, w: 0.06, h: 2.9, fill: { color: LIME } });
s.addText("GDN2", { x: rxc + 0.3, y: 2.38, w: cwid - 0.6, h: 0.3, fontFace: HF, fontSize: 12, color: LIME, charSpacing: 0.5, bold: true });
s.addText("Everything above, plus learned memory control:", { x: rxc + 0.3, y: 2.76, w: cwid - 0.6, h: 0.35, fontFace: BF, fontSize: 12.5, color: TXT, bold: true });
s.addText(["Fixed-state memory", "Channel-wise erase / write gates", "Cheaper to serve than a same-quality softmax model", "Stronger memory → references, editing, video"].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, color: BODY } })),
  { x: rxc + 0.3, y: 3.18, w: cwid - 0.6, h: 1.85, fontFace: BF, fontSize: 12.5, lineSpacingMultiple: 1.25, valign: "top" });
s.addShape(p.ShapeType.roundRect, { x: M, y: 5.35, w: W - 2 * M, h: 1.3, rectRadius: 0.06, fill: { color: CARD }, line: { color: LIME, width: 1.5 } });
s.addText([
  { text: "Why it matters:  ", options: { color: LIME, bold: true } },
  { text: "at equal quality, attention-light GDN2 serves cheaper than softmax — the gap widens at 4MP and holds as we scale to a ~2–3B production model. Plus erase / write gates for identity, background, and single-region edits. Cost edge benchmarked via VRAM, latency, and cost per image.", options: { color: TXT } },
], { x: M + 0.35, y: 5.35, w: W - 2 * M - 0.7, h: 1.3, fontFace: BF, fontSize: 13.5, valign: "middle", lineSpacingMultiple: 1.12 });
foot(s, 4);

// ===================================================== 5 PROPOSED SOLUTION (mapping)
s = p.addSlide(); bg(s);
title(s, [["An Attention-Light ", TXT], ["GDN2 Image Model", LIME]]);
s.addText("Each problem maps to a deliberate architecture choice — built to serve cheap, controllable image workflows.",
  { x: M, y: 1.55, w: 11.9, h: 0.5, fontFace: BF, fontSize: 15.5, color: MUT });
const maps = [
  ["High-res serving cost", "GDN2 state mixing replaces dense attention in most blocks → bounded memory, lower cost per megapixel."],
  ["Reference & edit cost", "One model, shared weights for generate / edit / inpaint / reference → no separate pipelines to run."],
  ["Control gap", "Instruction editing, masked inpainting, and reference conditioning are first-class API operations."],
];
maps.forEach((d, i) => {
  const y = 2.4 + i * 1.0;
  card(s, M, y, 3.55, 0.82, CARD2);
  s.addShape(p.ShapeType.rect, { x: M, y, w: 0.06, h: 0.82, fill: { color: ROSE } });
  s.addText(d[0], { x: M + 0.25, y, w: 3.2, h: 0.82, fontFace: HF, fontSize: 13.5, bold: true, color: TXT, valign: "middle" });
  s.addText("→", { x: M + 3.7, y, w: 0.6, h: 0.82, fontFace: BF, fontSize: 24, color: LIME, align: "center", valign: "middle", bold: true });
  card(s, M + 4.4, y, W - M - (M + 4.4), 0.82);
  s.addShape(p.ShapeType.rect, { x: M + 4.4, y, w: 0.06, h: 0.82, fill: { color: LIME } });
  s.addText(d[1], { x: M + 4.65, y, w: W - M - (M + 4.4) - 0.4, h: 0.82, fontFace: BF, fontSize: 13.5, color: BODY, valign: "middle", lineSpacingMultiple: 1.05 });
});
callout(s, 5.65, 0.95, [["The mechanism:  ", LIME, true], ["hybrid near-linear scaling with bounded exact-attention anchors — efficiency without giving up quality.", TXT]]);
foot(s, 5);

// ===================================================== 6 PROOF
s = p.addSlide(); bg(s);
title(s, [["The GDN2 Prototype — ", TXT], ["Trained on One GPU", LIME]]);
s.addShape(p.ShapeType.rect, { x: M - 0.04, y: 1.81, w: 4.28, h: 4.28, fill: { color: "060B16" }, line: { color: LINE, width: 1 } });
s.addImage({ path: "/tmp/boomer_deck/demo_gdn2.png", x: M, y: 1.85, w: 4.2, h: 4.2 });
s.addText("1024px (1MP) output from the prototype", { x: M, y: 6.1, w: 4.2, h: 0.3, fontFace: BF, fontSize: 10, color: MUT, align: "center", italic: true });
let rx = M + 4.8, rw = W - M - rx;
const st = [["~700M", "GDN2 params"], ["1024px", "1MP today"], ["35", "GPU-hours"], ["~$31", "compute"]];
let sw = (rw - 3 * 0.2) / 4;
st.forEach((d, i) => {
  const x = rx + i * (sw + 0.2);
  s.addText(d[0], { x, y: 1.85, w: sw + 0.2, h: 0.55, fontFace: HF, fontSize: 22, bold: true, color: LIME });
  s.addText(d[1], { x, y: 2.42, w: sw + 0.3, h: 0.3, fontFace: BF, fontSize: 10.5, color: MUT });
});
s.addText([
  "Single RTX PRO 6000 Blackwell GPU — no cluster.",
  "3.8M images @512px training, 600K @1024px fine-tune.",
  "32× DC-AE VAE · frozen Gemma 4 2B text encoder.",
].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 14 }, color: BODY } })),
  { x: rx, y: 3.05, w: rw, h: 1.35, fontFace: BF, fontSize: 13.5, lineSpacingMultiple: 1.28, valign: "top" });
s.addShape(p.ShapeType.roundRect, { x: rx, y: 4.5, w: rw, h: 0.7, rectRadius: 0.06, fill: { color: "20300F" }, line: { color: LIME, width: 1.25 } });
s.addText([{ text: "Next benchmark:  ", options: { color: LIME, bold: true } }, { text: "cost-per-4MP image vs FLUX.2 Klein.", options: { color: TXT } }],
  { x: rx + 0.25, y: 4.5, w: rw - 0.5, h: 0.7, fontFace: BF, fontSize: 13.5, valign: "middle" });
card(s, rx, 5.35, rw, 1.3, CARD2);
s.addText("HUGGING FACE MODEL", { x: rx + 0.25, y: 5.48, w: rw - 0.5, h: 0.28, fontFace: HF, fontSize: 9, color: LIME, charSpacing: 1, bold: true });
s.addText("huggingface.co/akrao9/BoomerV2-Text-to-Image", { x: rx + 0.25, y: 5.73, w: rw - 0.5, h: 0.3, fontFace: MF, fontSize: 11, color: TXT, hyperlink: { url: "https://huggingface.co/akrao9/BoomerV2-Text-to-Image" } });
s.addText("LIVE 1024px DEMO", { x: rx + 0.25, y: 6.06, w: rw - 0.5, h: 0.28, fontFace: HF, fontSize: 9, color: LIME, charSpacing: 1, bold: true });
s.addText("huggingface.co/spaces/akrao9/BoomerV2-T2I", { x: rx + 0.25, y: 6.31, w: rw - 0.5, h: 0.3, fontFace: MF, fontSize: 11, color: TXT, hyperlink: { url: "https://huggingface.co/spaces/akrao9/BoomerV2-T2I" } });
foot(s, 6);

// ===================================================== 7 MEASURED EFFICIENCY
s = p.addSlide(); bg(s);
title(s, [["Measured Efficiency: ", TXT], ["Flat Memory Scaling", LIME]]);
s.addText("Real numbers from the GDN2 prototype — serving memory is dominated by weights, not an O(N²) attention map.",
  { x: M, y: 1.55, w: 11.9, h: 0.5, fontFace: BF, fontSize: 15, color: MUT });
const estat = [["701M", "GDN2 params"], ["1024px", "output"], ["~1.5 GB", "VRAM (flat)"], ["121", "img/s batched"]];
let esw = (W - 2 * M - 3 * 0.2) / 4;
estat.forEach((d, i) => {
  const x = M + i * (esw + 0.2);
  s.addText(d[0], { x, y: 1.95, w: esw, h: 0.6, fontFace: HF, fontSize: 26, bold: true, color: LIME, align: "center" });
  s.addText(d[1], { x, y: 2.58, w: esw, h: 0.3, fontFace: MF, fontSize: 10.5, color: MUT, align: "center" });
});
const ecol = [2.2, 3.1, 3.2, 3.43];
tableRows(s, M, 2.98, ecol, [
  { cells: ["Batch", "Fwd latency", "Throughput (eager)", "Peak VRAM"], _h: 0.48 },
  { cells: ["1", "59 ms", "17 img/s", "1.49 GB"], _h: 0.5 },
  { cells: ["2", "59 ms", "34 img/s", "1.53 GB"], _h: 0.5 },
  { cells: ["4", "59 ms", "68 img/s", "1.62 GB"], _h: 0.5 },
  { cells: ["8", "88 ms", "90 img/s", "1.80 GB"], _h: 0.5 },
]);
s.addShape(p.ShapeType.roundRect, { x: M, y: 5.6, w: W - 2 * M, h: 1.0, rectRadius: 0.06, fill: { color: CARD }, line: { color: LIME, width: 1.5 } });
s.addText([
  { text: "8× batch → +21% memory (flat ~1.5 GB) — the linear-attention serving win.  ", options: { color: LIME, bold: true } },
  { text: "Single image: ~1.9 s eager. Batched + compiled: up to 121 img/s at batch 8. (Denoiser-only.)", options: { color: TXT } },
], { x: M + 0.35, y: 5.6, w: W - 2 * M - 0.7, h: 1.0, fontFace: BF, fontSize: 13.5, valign: "middle", lineSpacingMultiple: 1.12 });
foot(s, 7);

// ===================================================== 8 THE 4MP PRICE-TO-BEAT
s = p.addSlide(); bg(s);
title(s, [["The 4MP ", TXT], ["Price-to-Beat", LIME]]);
s.addText("FLUX.2 Klein 4B sets the current low-cost benchmark — Boomer targets a lower all-in cost.",
  { x: M, y: 1.55, w: 11.9, h: 0.5, fontFace: BF, fontSize: 15.5, color: MUT });
card(s, M, 2.1, 5.85, 3.05);
s.addText("FLUX.2 KLEIN 4B — OFFICIAL PRICING", { x: M + 0.3, y: 2.3, w: 5.3, h: 0.3, fontFace: HF, fontSize: 11, color: MUT, charSpacing: 0.5, bold: true });
[["$0.014", "first generated MP"], ["$0.001", "each additional MP"], ["$0.001 / MP", "reference image"]].forEach((d, i) => {
  const y = 2.78 + i * 0.74;
  s.addText(d[0], { x: M + 0.3, y, w: 2.6, h: 0.55, fontFace: HF, fontSize: 21, bold: true, color: TXT, valign: "middle" });
  s.addText(d[1], { x: M + 3.0, y, w: 2.7, h: 0.55, fontFace: BF, fontSize: 12.5, color: BODY, valign: "middle" });
});
s.addShape(p.ShapeType.roundRect, { x: M + 6.15, y: 2.1, w: 5.85, h: 3.05, rectRadius: 0.06, fill: { color: "20300F" }, line: { color: LIME, width: 1.75 } });
s.addText("4MP OUTPUT COST", { x: M + 6.45, y: 2.3, w: 5.2, h: 0.3, fontFace: HF, fontSize: 11, color: LIME, charSpacing: 0.5, bold: true });
s.addText("$0.014  +  3 × $0.001", { x: M + 6.45, y: 2.7, w: 5.2, h: 0.45, fontFace: BF, fontSize: 17, color: BODY, valign: "middle" });
s.addText("= $0.017", { x: M + 6.45, y: 3.12, w: 5.2, h: 0.95, fontFace: HF, fontSize: 46, bold: true, color: LIME, valign: "middle" });
s.addText("per 4MP image — the number to beat.", { x: M + 6.45, y: 4.15, w: 5.2, h: 0.3, fontFace: BF, fontSize: 12.5, italic: true, color: MUT });
s.addText([{ text: "Boomer target:  ", options: { color: TXT, bold: true } }, { text: "4MP below $0.017 · 1MP $0.005–0.010.", options: { color: TXT } }],
  { x: M + 6.45, y: 4.52, w: 5.2, h: 0.45, fontFace: BF, fontSize: 13.5, valign: "middle" });
s.addShape(p.ShapeType.roundRect, { x: M, y: 5.4, w: W - 2 * M, h: 1.2, rectRadius: 0.06, fill: { color: CARD }, line: { color: LIME, width: 1.5 } });
s.addText([
  { text: "Compete on all-in workflow cost:  ", options: { color: LIME, bold: true } },
  { text: "output + references + edits, not one per-MP number. Provider pricing varies (Fal lists $0.005/MP), so the official BFL $0.017 / 4MP is the public benchmark Boomer prices against.", options: { color: TXT } },
], { x: M + 0.35, y: 5.4, w: W - 2 * M - 0.7, h: 1.2, fontFace: BF, fontSize: 13.5, valign: "middle", lineSpacingMultiple: 1.12 });
foot(s, 8);

// ===================================================== 9 PHASE 1 PRODUCT (image API)
s = p.addSlide(); bg(s);
title(s, [["Phase 1: ", TXT], ["1MP Fast + 4MP Pro API", LIME]]);
s.addText("One focused product — a low-cost image generation & editing API built on a ~2–3B attention-light model.",
  { x: M, y: 1.5, w: 11.9, h: 0.45, fontFace: BF, fontSize: 15, color: MUT });
card(s, M, 2.05, 5.85, 2.05);
s.addText("1MP FAST", { x: M + 0.3, y: 2.23, w: 5, h: 0.35, fontFace: HF, fontSize: 16, color: LIME, bold: true });
s.addText(["1024×1024 generation", "lowest latency, lowest cost", "high-volume drafts & variants"].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, color: BODY } })),
  { x: M + 0.3, y: 2.65, w: 5.2, h: 1.3, fontFace: BF, fontSize: 13, lineSpacingMultiple: 1.25, valign: "top" });
card(s, M + 6.15, 2.05, 5.85, 2.05, CARD2);
s.addText("4MP PRO", { x: M + 6.45, y: 2.23, w: 5, h: 0.35, fontFace: HF, fontSize: 16, color: LIME, bold: true });
s.addText(["up to ~2048×2048 output", "production assets & product visuals", "editing, inpainting, reference workflows"].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, color: BODY } })),
  { x: M + 6.45, y: 2.65, w: 5.2, h: 1.3, fontFace: BF, fontSize: 13, lineSpacingMultiple: 1.25, valign: "top" });
s.addText("API ENDPOINTS", { x: M, y: 4.3, w: 5, h: 0.3, fontFace: HF, fontSize: 10, color: MUT, charSpacing: 1, bold: true });
const eps = ["/generate", "/edit", "/inpaint", "/reference", "/product"];
let ew = (W - 2 * M - 4 * 0.2) / 5;
eps.forEach((e, i) => {
  const x = M + i * (ew + 0.2);
  card(s, x, 4.65, ew, 0.7);
  s.addText(e, { x: x, y: 4.65, w: ew, h: 0.7, fontFace: MF, fontSize: 14, bold: true, color: LIME, align: "center", valign: "middle" });
});
callout(s, 5.65, 0.95, [["Goes straight at FLUX.2 Klein workflows — ", LIME, true], ["generation, editing, references, and 4MP output, priced on all-in cost. One product, shipped first.", TXT]]);
foot(s, 9);

// ===================================================== 10 PHASE 1 PIPELINE (Qwen VLM + GDN-2)
s = p.addSlide(); bg(s);
title(s, [["Phase 1 Plan: ", TXT], ["End-to-End Hybrid Pipeline", LIME]]);
s.addText("Phase 1 swaps the text encoder for a vision-language model — so Boomer reasons over the prompt and the reference image together.",
  { x: M, y: 1.55, w: 11.9, h: 0.5, fontFace: BF, fontSize: 15, color: MUT });
const bw = 5.4, bh = 2.15, by = 2.2;
const lx = M, gx = M + bw + 0.8;
card(s, lx, by, bw, bh, CARD2);
s.addShape(p.ShapeType.rect, { x: lx, y: by, w: 0.06, h: bh, fill: { color: LIME } });
s.addText("QWEN3.5 VLM", { x: lx + 0.3, y: by + 0.18, w: bw - 0.6, h: 0.35, fontFace: HF, fontSize: 15, color: LIME, bold: true });
s.addText("THE BRAIN — visual reasoning", { x: lx + 0.3, y: by + 0.55, w: bw - 0.6, h: 0.28, fontFace: MF, fontSize: 10, color: MUT, charSpacing: 0.5 });
s.addText(["Gated DeltaNet hybrid (≈75% linear)", "Sees prompt + reference image together", "Extracts identity, material, geometry"].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, color: BODY } })),
  { x: lx + 0.3, y: by + 0.95, w: bw - 0.6, h: 1.1, fontFace: BF, fontSize: 12.5, lineSpacingMultiple: 1.25, valign: "top" });
s.addShape(p.ShapeType.rightArrow, { x: lx + bw + 0.12, y: by + bh / 2 - 0.22, w: 0.56, h: 0.44, fill: { color: LIME }, line: { type: "none" } });
s.addText("embedding", { x: lx + bw + 0.06, y: by + bh / 2 + 0.3, w: 0.9, h: 0.25, fontFace: BF, fontSize: 8.5, color: MUT, align: "center" });
card(s, gx, by, bw, bh, CARD2);
s.addShape(p.ShapeType.rect, { x: gx, y: by, w: 0.06, h: bh, fill: { color: LIME } });
s.addText("BOOMER GDN-2", { x: gx + 0.3, y: by + 0.18, w: bw - 0.6, h: 0.35, fontFace: HF, fontSize: 15, color: LIME, bold: true });
s.addText("THE RENDERER — diffusion", { x: gx + 0.3, y: by + 0.55, w: bw - 0.6, h: 0.28, fontFace: MF, fontSize: 10, color: MUT, charSpacing: 0.5 });
s.addText(["Channel-wise erase / write gates", "Executes targeted edits at 4MP", "Keeps what matters, changes one region"].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, color: BODY } })),
  { x: gx + 0.3, y: by + 0.95, w: bw - 0.6, h: 1.1, fontFace: BF, fontSize: 12.5, lineSpacingMultiple: 1.25, valign: "top" });
card(s, M, 4.55, W - 2 * M, 0.7);
s.addText([
  { text: "EDGE: ", options: { color: LIME, bold: true } }, { text: "Qwen3.5 0.8B / 2B / 4B dense VLM        ", options: { color: TXT } },
  { text: "CLOUD: ", options: { color: LIME, bold: true } }, { text: "Qwen3.5 9B (or A3B MoE) for pro multi-image workflows", options: { color: TXT } },
], { x: M + 0.3, y: 4.55, w: W - 2 * M - 0.6, h: 0.7, fontFace: BF, fontSize: 13, valign: "middle" });
s.addShape(p.ShapeType.roundRect, { x: M, y: 5.45, w: W - 2 * M, h: 1.1, rectRadius: 0.06, fill: { color: CARD }, line: { color: LIME, width: 1.5 } });
s.addText([
  { text: "Both stages are Gated DeltaNet hybrids — ", options: { color: LIME, bold: true } },
  { text: "strong visual reasoning + high-res rendering at low serving cost. Phase 1 upgrade: today's prototype uses a frozen Gemma text encoder.", options: { color: TXT } },
], { x: M + 0.35, y: 5.45, w: W - 2 * M - 0.7, h: 1.1, fontFace: BF, fontSize: 13.5, valign: "middle", lineSpacingMultiple: 1.12 });
foot(s, 10);

// ===================================================== 11 ROADMAP
s = p.addSlide(); bg(s);
title(s, [["From Prototype to ", TXT], ["Unified Worlds", LIME]]);
const phases = [
  ["NOW", "~700M GDN2 1024px prototype — single-GPU proof, public demo"],
  ["PHASE 1", "~2–3B image model — 1MP Fast + 4MP Pro API, edit & reference"],
  ["PHASE 2", "Boomer Avatar video + unified GDN2 MMDiT"],
  ["PHASE 3", "Larger image-to-video / text-to-video"],
  ["LONG-TERM", "Interactive world models"],
];
let ty = 1.95;
phases.forEach((d, i) => {
  const y = ty + i * 0.84;
  const on = i === 0;
  s.addShape(p.ShapeType.ellipse, { x: M + 0.02, y: y + 0.06, w: 0.24, h: 0.24, fill: { color: on ? LIME : CARD2 }, line: { color: on ? LIME : LINE, width: 1.5 } });
  if (i < phases.length - 1) s.addShape(p.ShapeType.line, { x: M + 0.14, y: y + 0.3, w: 0, h: 0.54, line: { color: LINE, width: 1.5 } });
  s.addText(d[0], { x: M + 0.5, y: y, w: 1.95, h: 0.4, fontFace: HF, fontSize: 12, bold: true, color: on ? LIME : MUT, valign: "middle" });
  s.addText(d[1], { x: M + 2.5, y: y - 0.02, w: 4.6, h: 0.5, fontFace: BF, fontSize: 13, color: BODY, valign: "middle", lineSpacingMultiple: 1.0 });
});
s.addShape(p.ShapeType.rect, { x: 7.62, y: 1.9, w: 5.05, h: 3.65, fill: { color: "000000" }, line: { color: LINE, width: 1 } });
s.addImage({ path: "/tmp/boomer_deck/world_model.png", x: 7.66, y: 1.94, w: 4.97, h: 3.57 });
s.addText("Long-term: one model, unlimited worlds.", { x: 7.62, y: 5.65, w: 5.05, h: 0.35, fontFace: BF, fontSize: 11, color: MUT, align: "center", italic: true });
foot(s, 11);

// ===================================================== 12 COMPETITIVE LANDSCAPE
s = p.addSlide(); bg(s);
title(s, [["Competitive ", TXT], ["Landscape", LIME]]);
const ccol = [2.95, 4.55, 4.43];
tableRows(s, M, 1.8, ccol, [
  { cells: ["Model", "Public position", "Boomer response"], _h: 0.55 },
  { cells: ["FLUX.2 Klein 4B", "Compact gen / edit / reference, official 4MP pricing", "Undercut 4MP serving cost via attention-light GDN2"], _h: 0.92 },
  { cells: ["FLUX.2 Pro / Max", "Higher-fidelity production tier", "Not the Phase 1 target"], _h: 0.82 },
  { cells: ["SD3 / MMDiT", "Validates the multimodal diffusion-transformer direction", "Phase 2 moves to a GDN2 MMDiT"], _h: 0.82 },
  { cells: ["Sana / PixArt-Σ", "Validates small, efficient high-res models", "Supports the efficient-model thesis"], _h: 0.82 },
  { cells: ["Boomer", "~700M prototype proven → ~2–3B Phase 1 model", "1MP Fast + 4MP Pro API at lower all-in cost"], _h: 0.92 },
], { boomRow: 5 });
foot(s, 12);

// ===================================================== 13 FOUNDER
s = p.addSlide(); bg(s);
title(s, [["The ", TXT], ["Founder", LIME]]);
s.addText("Ankamma Rao Bandi", { x: M, y: 1.5, w: 11.9, h: 0.6, fontFace: HF, fontSize: 30, bold: true, color: TXT });
s.addText("Solo Founder · Software Engineer · MS, Computer Science", { x: M, y: 2.12, w: 11.9, h: 0.35, fontFace: BF, fontSize: 14.5, color: LIME });
const fcw = (W - 2 * M - 0.3) / 2;
card(s, M, 2.65, fcw, 2.25);
s.addText("BACKGROUND", { x: M + 0.3, y: 2.83, w: fcw - 0.6, h: 0.3, fontFace: HF, fontSize: 11, color: MUT, charSpacing: 1, bold: true });
s.addText(["Generative vision & multimodal AI", "Efficient sub-quadratic / linear-attention architectures", "Designs custom attention to cut compute & memory of diffusion models"].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, color: BODY } })),
  { x: M + 0.3, y: 3.22, w: fcw - 0.6, h: 1.55, fontFace: BF, fontSize: 13, lineSpacingMultiple: 1.3, valign: "top" });
const fx2 = M + fcw + 0.3;
card(s, fx2, 2.65, fcw, 2.25, CARD2);
s.addShape(p.ShapeType.rect, { x: fx2, y: 2.65, w: 0.06, h: 2.25, fill: { color: LIME } });
s.addText("WHAT I'VE BUILT", { x: fx2 + 0.3, y: 2.83, w: fcw - 0.6, h: 0.3, fontFace: HF, fontSize: 11, color: LIME, charSpacing: 1, bold: true });
s.addText(["Boomer V2 end-to-end — data, model, training, inference", "~700M GDN2 model trained from scratch on a single GPU", "Live on Hugging Face: public weights + interactive demo"].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, color: BODY } })),
  { x: fx2 + 0.3, y: 3.22, w: fcw - 0.6, h: 1.55, fontFace: BF, fontSize: 13, lineSpacingMultiple: 1.3, valign: "top" });
s.addShape(p.ShapeType.roundRect, { x: M, y: 5.1, w: W - 2 * M, h: 1.3, rectRadius: 0.06, fill: { color: CARD }, line: { color: LIME, width: 1.5 } });
s.addText([
  { text: "Why me:  ", options: { color: LIME, bold: true } },
  { text: "I turned a sub-quadratic research insight into a working, public model — alone, for ~$31 of compute. The raise funds the first ML / infra hire to scale from prototype to a production API.", options: { color: TXT } },
], { x: M + 0.35, y: 5.1, w: W - 2 * M - 0.7, h: 1.3, fontFace: BF, fontSize: 14, valign: "middle", lineSpacingMultiple: 1.12 });
foot(s, 13);

// ===================================================== 14 ASK
s = p.addSlide(); bg(s);
title(s, [["The ", TXT], ["Pre-Seed", LIME]]);
s.addText([
  { text: "The pre-seed funds our first flagship ~2–3B image generation + edit model. ", options: { color: BODY } },
  { text: "Estimated training compute: ~$20-25K", options: { color: LIME, bold: true } },
  { text: "  — the rest funds data, infra, and go-to-market.", options: { color: BODY } },
], { x: M, y: 1.44, w: 11.9, h: 0.5, fontFace: BF, fontSize: 13.5, valign: "middle", lineSpacingMultiple: 1.1 });
card(s, M, 2.0, 6.1, 4.45);
s.addText("USE OF FUNDS", { x: M + 0.3, y: 2.2, w: 5, h: 0.3, fontFace: HF, fontSize: 11, color: LIME, charSpacing: 1, bold: true });
const funds = ["Compute — image model training + 4MP fine-tune", "Image, reference & editing datasets", "API & inference infrastructure", "Founder runway + first ML / infra hire", "Customer pilots, benchmarks & evals"];
funds.forEach((d, i) => {
  const y = 2.7 + i * 0.72;
  s.addShape(p.ShapeType.rect, { x: M + 0.3, y: y + 0.1, w: 0.12, h: 0.12, fill: { color: LIME }, line: { type: "none" } });
  s.addText(d, { x: M + 0.62, y, w: 5.25, h: 0.5, fontFace: BF, fontSize: 14, color: BODY, valign: "middle", lineSpacingMultiple: 1.05 });
  if (i < funds.length - 1) s.addShape(p.ShapeType.line, { x: M + 0.3, y: y + 0.62, w: 5.5, h: 0, line: { color: LINE, width: 0.5 } });
});
card(s, M + 6.4, 2.0, 5.6, 4.45, CARD2);
s.addText("MILESTONES", { x: M + 6.7, y: 2.2, w: 5, h: 0.3, fontFace: HF, fontSize: 11, color: LIME, charSpacing: 1, bold: true });
s.addText([
  "Train ~2–3B Boomer Image model",
  "Ship 1MP Fast + 4MP Pro API",
  "Add edit / inpaint / reference workflows",
  "Benchmark quality + cost vs FLUX.2 Klein",
  "Convert 10–15 buyer convos → 3–5 design partners",
  "First paid API pilots",
  "Hire first ML / infra engineer",
  "Scope Boomer Avatar (Phase 2)",
].map(t => ({ text: t, options: { bullet: { code: "2022", indent: 14 }, color: BODY } })),
  { x: M + 6.7, y: 2.65, w: 5.0, h: 3.7, fontFace: BF, fontSize: 13, lineSpacingMultiple: 1.32, valign: "top" });
foot(s, 14);

p.writeFile({ fileName: "/tmp/boomer_deck/Boomer_AI_Deck4.pptx" }).then(f => console.log("WROTE", f));
