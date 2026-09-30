// Shared pieces of the Armenia treatise plates:
//  - Engraver: a fullscreen pass that turns a footage frame into a line engraving (bone lines on ink,
//    or ink lines on bone "paper"); the lines bend with the tone like a burin following a form, deep
//    tones get a cross-hatch, and the footage's signal mask (orange clothes, sun glints) is printed in
//    the palette's orange (it blooms).
//  - TerrainMap: the contour-engraved relief of the Yerevan–Sevan region (analysis/make_map.py).
//  - PVal / drawReadout: the P(восторг) instrument ("probability of delight"), staged in the plates.
//  - Type helpers for the treatise voice: table numbers, captions, deadpan footnotes.
import * as THREE from 'three';
import { FSPass, W, H } from '../engine/gl';
import { rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, noise1, prog, smoothstep } from '../engine/util';

// ------------------------------------------------------------------ engraving
/**
 * The book-plate vignette: the engraving thins out to bare ground above and below the picture, along a
 * slightly irregular edge (a hand-cut plate), leaving clean ground for the instrument and the caption.
 * `mask` = (top fade start, top fade end, bottom fade start, bottom fade end) in screen height from the top.
 */
const PLATE_MASK_GLSL = /* glsl */ `
// hatch() keeps an anti-aliasing hairline even at zero darkness (grey moire over bare ground); this one
// box-filters the line so no darkness means no line
float hatchB(float u, float darkness) {
  float x = abs(fract(u + 0.5) - 0.5);
  float hw = 0.5 * sat(darkness), aa = max(fwidth(u), 1e-4) * 0.75;
  return min(_boxLine(x, hw, aa) + _boxLine(1.0 - x, hw, aa), 1.0);
}
float plateMask(vec2 frag, vec4 mask) {
  float y = 1.0 - frag.y / ${H.toFixed(1)} + 0.012 * snoise(vec2(frag.x * 0.006, 3.7)) + 0.005 * snoise(vec2(frag.x * 0.03, 9.1));
  return sat((y - mask.x) / max(mask.y - mask.x, 1e-4)) * (1.0 - sat((y - mask.z) / max(mask.w - mask.z, 1e-4)));
}`;
/** Plates: the picture between ~0.11 and ~0.70 of the height. */
export const PLATE_MASK: [number, number, number, number] = [0.105, 0.17, 0.62, 0.72];
export const NO_MASK: [number, number, number, number] = [-1, -0.9, 1.9, 2];

export interface EngraveParams {
  zoom?: number; rot?: number; panX?: number; panY?: number;
  paper?: number; freq?: number; angle?: number; gain?: number; lift?: number; gamma?: number;
  sig?: number; bend?: number; fade?: number; reveal?: number; revealSoft?: number;
  /** Inset window [x, y, w, h] in px from the top-left (the footage fills it); omit for full frame. */
  win?: [number, number, number, number];
  mask?: [number, number, number, number];
}

export class Engraver {
  pass = new FSPass(/* glsl */ `
    uniform sampler2D tex;
    uniform vec4 xf;          // zoom, rotation, pan x, pan y (uv units)
    uniform vec4 win;         // inset window in px (x, y from the top-left, w, h); w = 0: full frame
    uniform vec4 mask;
    ${PLATE_MASK_GLSL}
    uniform float paper, freq, angle, gain, lift, gam, sig, bend, fade, reveal, revealSoft;
    void main() {
      vec2 frag = FRAG_PX;
      vec2 uv = frag / vec2(${W.toFixed(1)}, ${H.toFixed(1)});
      float asp = ${(W / H).toFixed(5)};
      if (win.z > 0.0) {
        vec2 ft = vec2(frag.x, ${H.toFixed(1)} - frag.y) - win.xy;
        if (ft.x < 0.0 || ft.y < 0.0 || ft.x > win.z || ft.y > win.w) discard;
        uv = vec2(ft.x / win.z, 1.0 - ft.y / win.w);
        asp = win.z / win.w;
      }
      vec2 c = uv - 0.5;
      c.x *= asp;
      c = rot2(xf.y) * c;
      c.x /= asp;
      // a window of another aspect crops the 9:16 footage (cover fit)
      c *= vec2(min(1.0, asp / ${(W / H).toFixed(5)}), min(1.0, ${(W / H).toFixed(5)} / asp));
      vec2 suv = c / xf.x + 0.5 + xf.zw;
      suv.y = 1.0 - suv.y;
      vec2 s = texture(tex, clamp(suv, vec2(0.001), vec2(0.999))).rg;
      float L = pow(clamp((s.r - lift) * gain, 0.0, 1.0), gam);
      float pm = plateMask(frag, mask);
      float cover = mix(L * 0.96, 1.0 - L, paper) * pm;
      float u = (rot2(angle) * frag).y * freq + bend * L;
      float a = hatchB(u, cover * 1.02);
      float u2 = (rot2(angle + 1.15) * frag).y * freq * 0.92 + bend * 0.6 * L;
      float b = hatchB(u2, sat(cover * 2.6 - 1.9));
      float ink = max(a, b);
      vec3 base = mix(C_INK, C_BONE, paper);
      vec3 line = mix(C_BONE * 0.9, C_INK, paper);
      vec3 col = mix(base, line, ink);
      float sg = smoothstep(0.22, 0.8, s.g) * sig * pm;
      vec3 hot = mix(heat(0.42 + 0.24 * s.g) * mix(0.55, 1.3, ink), C_SIGNAL * mix(0.85, 0.6, ink), paper);
      col = mix(col, hot, sg);
      // the burin: the engraving is cut from the top down, its edge a soft band
      float edge = 1.0 - uv.y;
      float shown = 1.0 - smoothstep(reveal * (1.0 + revealSoft) - revealSoft, reveal * (1.0 + revealSoft), edge);
      col = mix(base, col, shown);
      col = mix(col, base, fade);
      fragColor = vec4(col, 1.0);
    }`, {
    tex: { value: null }, xf: { value: new THREE.Vector4(1, 0, 0, 0) }, win: { value: new THREE.Vector4(0, 0, 0, 0) },
    mask: { value: new THREE.Vector4(...NO_MASK) },
    paper: { value: 0 }, freq: { value: 0.14 }, angle: { value: 0.35 }, gain: { value: 1.1 }, lift: { value: 0.04 },
    gam: { value: 1.1 }, sig: { value: 1 }, bend: { value: 0.7 }, fade: { value: 0 }, reveal: { value: 1 }, revealSoft: { value: 0.08 },
  });

  render(renderer: THREE.WebGLRenderer, out: THREE.WebGLRenderTarget, tex: THREE.Texture, p: EngraveParams = {}) {
    const u = this.pass.u;
    u.tex!.value = tex;
    (u.xf!.value as THREE.Vector4).set(p.zoom ?? 1, p.rot ?? 0, p.panX ?? 0, p.panY ?? 0);
    (u.win!.value as THREE.Vector4).set(...(p.win ?? [0, 0, 0, 0]));
    (u.mask!.value as THREE.Vector4).set(...(p.mask ?? NO_MASK));
    u.paper!.value = p.paper ?? 0;
    u.freq!.value = p.freq ?? 0.14;
    u.angle!.value = p.angle ?? 0.35;
    u.gain!.value = p.gain ?? 1.1;
    u.lift!.value = p.lift ?? 0.04;
    u.gam!.value = p.gamma ?? 1.1;
    u.sig!.value = p.sig ?? 1;
    u.bend!.value = p.bend ?? 0.7;
    u.fade!.value = p.fade ?? 0;
    u.reveal!.value = p.reveal ?? 1;
    u.revealSoft!.value = p.revealSoft ?? 0.08;
    this.pass.render(renderer, out);
  }
}

// ------------------------------------------------------------------ natural footage
export interface PhotoParams {
  zoom?: number; rot?: number; panX?: number; panY?: number;
  win?: [number, number, number, number];
  /** 0..1 how far the photograph has developed from the top (of the window) down; below it the pass leaves
   *  what is already in the target (the engraving of the same frame). 1 (default) = all of it. */
  develop?: number;
  fade?: number; contrast?: number; saturation?: number; exposure?: number;
}

/** The footage in natural colour (sRGB JPEG frames, decoded to linear), with the Engraver's framing. */
export class Photo {
  pass = new FSPass(/* glsl */ `
    uniform sampler2D tex;
    uniform vec4 xf, win;
    uniform float develop, fade, contrast, satur, expo;
    void main() {
      vec2 frag = FRAG_PX;
      vec2 uv = frag / vec2(${W.toFixed(1)}, ${H.toFixed(1)});
      float asp = ${(W / H).toFixed(5)};
      if (win.z > 0.0) {
        vec2 ft = vec2(frag.x, ${H.toFixed(1)} - frag.y) - win.xy;
        if (ft.x < 0.0 || ft.y < 0.0 || ft.x > win.z || ft.y > win.w) discard;
        uv = vec2(ft.x / win.z, 1.0 - ft.y / win.w);
        asp = win.z / win.w;
      }
      if (1.0 - uv.y > develop) discard;
      vec2 c = uv - 0.5;
      c.x *= asp;
      c = rot2(xf.y) * c;
      c.x /= asp;
      c *= vec2(min(1.0, asp / ${(W / H).toFixed(5)}), min(1.0, ${(W / H).toFixed(5)} / asp));
      vec2 suv = c / xf.x + 0.5 + xf.zw;
      suv.y = 1.0 - suv.y;
      vec3 col = texture(tex, clamp(suv, vec2(0.001), vec2(0.999))).rgb * expo;
      // a gentle grade: contrast about mid-grey (in a perceptual space), saturation
      vec3 p = pow(max(col, 0.0), vec3(1.0 / 2.2));
      p = (p - 0.5) * contrast + 0.5;
      p = mix(vec3(dot(p, vec3(0.2126, 0.7152, 0.0722))), p, satur);
      col = pow(max(p, 0.0), vec3(2.2));
      col = mix(col, C_INK, fade);
      fragColor = vec4(col, 1.0);
    }`, {
    tex: { value: null }, xf: { value: new THREE.Vector4(1, 0, 0, 0) }, win: { value: new THREE.Vector4(0, 0, 0, 0) },
    develop: { value: 1 }, fade: { value: 0 }, contrast: { value: 1.04 }, satur: { value: 1.06 }, expo: { value: 1 },
  });

  render(renderer: THREE.WebGLRenderer, out: THREE.WebGLRenderTarget, tex: THREE.Texture, p: PhotoParams = {}) {
    const u = this.pass.u;
    u.tex!.value = tex;
    (u.xf!.value as THREE.Vector4).set(p.zoom ?? 1, p.rot ?? 0, p.panX ?? 0, p.panY ?? 0);
    (u.win!.value as THREE.Vector4).set(...(p.win ?? [0, 0, 0, 0]));
    u.develop!.value = p.develop ?? 1.01;
    u.fade!.value = p.fade ?? 0;
    u.contrast!.value = p.contrast ?? 1.04;
    u.satur!.value = p.saturation ?? 1.06;
    u.expo!.value = p.exposure ?? 1;
    this.pass.render(renderer, out);
  }
}

/** Ink gradients that keep the type legible over natural footage: top band and bottom band (px). */
export function scrims(c: CanvasRenderingContext2D, o: { top?: [number, number, number]; bottom?: [number, number, number] } = {}) {
  c.save();
  c.shadowBlur = 0;
  if (o.bottom) {
    const [y0, y1, a] = o.bottom;
    const g = c.createLinearGradient(0, y0, 0, y1);
    g.addColorStop(0, rgba('ink', 0));
    g.addColorStop(0.55, rgba('ink', a * 0.72));
    g.addColorStop(1, rgba('ink', a));
    c.fillStyle = g;
    c.fillRect(0, y0, W, y1 - y0);
    c.fillStyle = rgba('ink', a);
    c.fillRect(0, y1, W, H - y1);
  }
  if (o.top) {
    const [y0, y1, a] = o.top;
    const g = c.createLinearGradient(0, y0, 0, y1);
    g.addColorStop(0, rgba('ink', a));
    g.addColorStop(0.55, rgba('ink', a * 0.8));
    g.addColorStop(1, rgba('ink', 0));
    c.fillStyle = g;
    c.fillRect(0, y0, W, y1 - y0);
  }
  c.restore();
}

// ------------------------------------------------------------------ terrain map
export interface MapMeta { n: number; bbox: [number, number, number, number]; km: [number, number]; places: Record<string, [number, number]> }

let MAP: { meta: MapMeta; tex: THREE.DataTexture } | null = null;
export async function loadMap() {
  if (MAP) return MAP;
  const meta: MapMeta = await (await fetch('map/map.json')).json();
  const src = new Uint16Array(await (await fetch('map/terrain.bin')).arrayBuffer());
  const half = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) half[i] = THREE.DataUtils.toHalfFloat(src[i]! / 65535);
  const tex = new THREE.DataTexture(half, meta.n, meta.n, THREE.RGFormat, THREE.HalfFloatType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  MAP = { meta, tex };
  return MAP;
}

/** Map camera: centre (map 0..1, y down), px per map unit, rotation. */
export interface MapCam { cx: number; cy: number; scale: number; rot?: number }

export function mapToScreen(cam: MapCam, mx: number, my: number) {
  const r = cam.rot ?? 0, dx = (mx - cam.cx) * cam.scale, dy = (my - cam.cy) * cam.scale;
  return { x: W / 2 + dx * Math.cos(r) - dy * Math.sin(r), y: H / 2 + dx * Math.sin(r) + dy * Math.cos(r) };
}

/** Contour engraving of the relief: 50 m contours, index contours every 250 m, the lake in water lines. */
export class TerrainMap {
  pass = new FSPass(/* glsl */ `
    uniform sampler2D tex;
    uniform vec4 cam;         // cx, cy (map 0..1, y down), px per map unit, rotation
    uniform float paper, fade, lit;
    uniform vec4 mask;
    ${PLATE_MASK_GLSL}
    float elevAt(vec2 m) { return texture(tex, vec2(m.x, 1.0 - m.y)).r * 3400.0 + 700.0; }
    void main() {
      vec2 p = vec2(FRAG_PX.x, ${H.toFixed(1)} - FRAG_PX.y);           // y down
      vec2 d = p - vec2(${(W / 2).toFixed(1)}, ${(H / 2).toFixed(1)});
      float r = -cam.w;
      d = mat2(cos(r), sin(r), -sin(r), cos(r)) * d;
      vec2 m = vec2(cam.x, cam.y) + d / cam.z;
      vec2 s = texture(tex, vec2(m.x, 1.0 - m.y)).rg;
      float e = s.r * 3400.0 + 700.0;
      // contour lines as distance to the level set: |e - k*step| / |grad e|
      float stepM = 100.0;
      float g = length(vec2(dFdx(e), dFdy(e))) + 1e-3;
      float k = e / stepM;
      float dmin = abs(fract(k + 0.5) - 0.5) * stepM / g;
      float isIndex = step(abs(fract(k / 5.0 + 0.1) - 0.1), 0.101);
      float line = 1.0 - smoothstep(0.35, 1.25, dmin);
      float idx = 1.0 - smoothstep(0.7, 1.8, dmin);
      float land = 1.0 - smoothstep(0.35, 0.6, s.g);
      float inside = step(0.0, m.x) * step(m.x, 1.0) * step(0.0, m.y) * step(m.y, 1.0);
      // hill shading as sparse hatch on the slopes (light from the north-west)
      vec2 gr = vec2(texture(tex, vec2(m.x + 0.002, 1.0 - m.y)).r - texture(tex, vec2(m.x - 0.002, 1.0 - m.y)).r,
                     texture(tex, vec2(m.x, 1.0 - m.y - 0.002)).r - texture(tex, vec2(m.x, 1.0 - m.y + 0.002)).r);
      float shade = sat(dot(normalize(vec3(-gr * 40.0, 1.0)), normalize(vec3(-0.6, -0.6, 0.55))));
      float hat = hatchB((rot2(0.8) * FRAG_PX).y * 0.16, sat(0.75 - shade) * 0.55 * lit);
      float water = s.g;
      float wl = hatchB(FRAG_PX.y * 0.13, 0.22) * water;
      vec3 base = mix(C_INK, C_BONE, paper);
      vec3 ink = mix(C_BONE, C_INK, paper);
      float a = land * inside * (line * mix(0.2, 0.55, isIndex) + idx * isIndex * 0.2 + hat * 0.12);
      a += inside * (wl * 0.55 + (1.0 - smoothstep(0.35, 1.0, abs(s.g - 0.5) * 60.0 / (length(vec2(dFdx(s.g), dFdy(s.g))) * 60.0 + 1e-3))) * 0.0);
      // lake shore
      float sg = length(vec2(dFdx(water), dFdy(water))) + 1e-4;
      float shore = 1.0 - smoothstep(0.6, 1.6, abs(water - 0.5) / sg);
      a += shore * inside * 0.8;
      vec3 col = mix(base, ink, clamp(a, 0.0, 1.0) * plateMask(FRAG_PX, mask));
      col = mix(col, base, fade);
      fragColor = vec4(col, 1.0);
    }`, { tex: { value: null }, cam: { value: new THREE.Vector4(0.5, 0.5, 1000, 0) }, paper: { value: 0 }, fade: { value: 0 }, lit: { value: 1 }, mask: { value: new THREE.Vector4(...NO_MASK) } });

  render(renderer: THREE.WebGLRenderer, out: THREE.WebGLRenderTarget, tex: THREE.Texture, cam: MapCam, o: { paper?: number; fade?: number; lit?: number; mask?: [number, number, number, number] } = {}) {
    const u = this.pass.u;
    (u.mask!.value as THREE.Vector4).set(...(o.mask ?? NO_MASK));
    u.tex!.value = tex;
    (u.cam!.value as THREE.Vector4).set(cam.cx, cam.cy, cam.scale, cam.rot ?? 0);
    u.paper!.value = o.paper ?? 0;
    u.fade!.value = o.fade ?? 0;
    u.lit!.value = o.lit ?? 1;
    this.pass.render(renderer, out);
  }
}

// ------------------------------------------------------------------ P(восторг)
/** Steps of the probability of delight through the day: [time, value]. */
export const DELIGHT: [number, number][] = [
  [-1, 0.02], [6.0, 0.15], [10.0, 0.42], [14.0, 0.61], [18.0, 0.64], [22.0, 0.81],
  [28.0, 0.93], [29.5, 0.97], [32.0, 0.99], [40.0, 1.0],
];

export class PVal {
  constructor(public steps: [number, number][] = DELIGHT) {}
  value(t: number): number {
    let i = 0;
    while (i + 1 < this.steps.length && this.steps[i + 1]![0] <= t) i++;
    const [t1, v1] = this.steps[i]!, v0 = this.steps[Math.max(0, i - 1)]![1];
    const k = i === 0 ? 1 : prog(t, t1, t1 + 0.8, ease.outExpo);
    const next = this.steps[i + 1];
    // the riser before the drop: the value creeps up (never more than 60% of the gap)
    const creep = next ? (next[1] - v1) * 0.6 * smoothstep(t1 + 0.6, next[0], t) * (next[0] === 22.0 ? 1 : 0.25) : 0;
    return clamp(v0 + (v1 - v0) * k + creep + noise1(t * 3.1, 7) * 0.003, 0, 1.2);
  }
  /** 0..1 flash right after a step. */
  flash(t: number) {
    let f = 0;
    for (const [s] of this.steps) if (s > 0 && t >= s) f = Math.max(f, Math.pow(0.5, (t - s) / 0.3));
    return f;
  }
}

export const fmtP = (v: number) => (v >= 0.995 && v < 1 ? v.toFixed(3) : v.toFixed(2));

/**
 * The instrument, drawn anywhere: small label, digits, a 0..1 tick bar with the value filled in
 * signal. (x, y) = left end of the digits' baseline. `ink` for bone-paper plates.
 */
export function drawReadout(c: CanvasRenderingContext2D, x: number, y: number, v: number, o: { scale?: number; ink?: boolean; flash?: number; label?: string; text?: string; barW?: number; alpha?: number } = {}) {
  const k = o.scale ?? 1, a = o.alpha ?? 1;
  const fg = o.ink ? 'ink' : 'bone';
  c.save();
  c.globalAlpha *= a;
  c.textBaseline = 'alphabetic';
  c.font = font(F.mono(500), 17 * k);
  c.letterSpacing = `${3 * k}px`;
  c.fillStyle = rgba(fg, 0.62);
  c.fillText(o.label ?? 'P(ВОСТОРГ)', x, y - 50 * k);
  c.letterSpacing = '0px';
  c.font = font(F.mono(400), 52 * k);
  const fl = clamp(o.flash ?? 0);
  c.fillStyle = fl > 0.02 ? rgba('signal', 0.65 + 0.35 * fl) : rgba(fg, 0.94);
  c.fillText(o.text ?? fmtP(v), x - 2 * k, y);
  const bw = (o.barW ?? 250) * k, by = y + 22 * k;
  c.fillStyle = rgba(fg, 0.3);
  c.fillRect(x, by, bw, 1.2 * k);
  for (let i = 0; i <= 10; i++) c.fillRect(x + (bw * i) / 10, by - (i % 5 === 0 ? 7 : 4) * k, 1.2 * k, (i % 5 === 0 ? 7 : 4) * k);
  c.fillStyle = rgba('signal', 0.95);
  c.fillRect(x, by - 1.5 * k, bw * clamp(v, 0, 1), 3.5 * k);
  if (v > 1) {  // the bar breaks its end cap
    c.fillRect(x + bw, by - 1.5 * k, bw * Math.min(v - 1, 0.5) * 6, 3.5 * k);
  }
  c.restore();
}

// ------------------------------------------------------------------ treatise type
let HALO: string | null = null;
/** Set typed labels on a chip of this colour (legible over bright footage); null = off. */
export function setHalo(color: string | null) { HALO = color; }

/** Type one line on: characters appear in order over `dur` seconds from t0 (no easing: a typewriter). */
export function typeOn(c: CanvasRenderingContext2D, s: string, x: number, y: number, t: number, t0: number, dur: number) {
  const n = Math.floor(clamp((t - t0) / Math.max(dur, 1e-3)) * s.length + 1e-6);
  if (n <= 0) return 0;
  if (HALO) {
    // the chip spans the whole line from the first character, so it doesn't jitter as the text types on
    const m = c.measureText(s), mn = c.measureText(s.slice(0, n));
    const al = c.textAlign, left = al === 'right' || al === 'end' ? x - m.width : al === 'center' ? x - m.width / 2 : x;
    const w = al === 'right' || al === 'end' || al === 'center' ? m.width : mn.width;
    c.save();
    c.shadowBlur = 0;
    c.fillStyle = HALO;
    c.fillRect(left - 7, y - mn.fontBoundingBoxAscent - 3, w + 14, mn.fontBoundingBoxAscent + mn.fontBoundingBoxDescent + 6);
    c.restore();
  }
  c.fillText(s.slice(0, n), x, y);
  return n;
}

/**
 * The plate's caption block (bottom-left, clear of the stories UI): table number in mono, the title in
 * Cormorant italic, a deadpan footnote in mono. Everything types on from t0.
 */
export function drawCaption(c: CanvasRenderingContext2D, t: number, t0: number, o: { table: string; title: string; notes: string[]; x?: number; y?: number; ink?: boolean; alpha?: number; size?: number }) {
  const x = o.x ?? 72, y = o.y ?? 1500, fg = o.ink ? 'ink' : 'bone', a = o.alpha ?? 1, size = o.size ?? 74;
  if (t < t0) return;
  c.save();
  c.globalAlpha *= a;
  c.textBaseline = 'alphabetic';
  c.font = font(F.mono(500), 20);
  c.letterSpacing = '5px';
  c.fillStyle = rgba(fg, 0.6);
  typeOn(c, o.table, x, y - size * 1.1, t, t0, 0.25);
  c.letterSpacing = '0px';
  // hairline rule under the table number
  const rl = 150 * ease.outExpo(clamp((t - t0 - 0.05) / 0.5));
  c.fillRect(x, y - size * 1.1 + 14, rl, 1.2);
  c.font = font(F.serif(600, true), size);
  c.fillStyle = rgba(fg, 0.97);
  const tp = ease.outExpo(clamp((t - t0 - 0.1) / 0.55));
  c.save();
  c.beginPath();
  c.rect(x - 20, y - size * 1.05, W, size * 1.35);
  c.clip();
  c.fillText(o.title, x, y + (1 - tp) * size * 0.9);
  c.restore();
  c.font = font(F.mono(400), 22);
  c.fillStyle = rgba(fg, 0.78);
  o.notes.forEach((s, i) => typeOn(c, s, x, y + 46 + i * 32, t, t0 + 0.35 + i * 0.22, 0.45));
  c.restore();
}

/** Small mono label with a leader line to a point (callout). */
export function callout(c: CanvasRenderingContext2D, px: number, py: number, lx: number, ly: number, text: string, t: number, t0: number, o: { ink?: boolean; signal?: boolean } = {}) {
  if (t < t0) return;
  const k = ease.outExpo(clamp((t - t0) / 0.4));
  const fg = o.signal ? 'signal' : o.ink ? 'ink' : 'bone';
  c.save();
  c.strokeStyle = rgba(fg, 0.8);
  c.fillStyle = rgba(fg, 0.9);
  c.lineWidth = 1.3;
  const ex = px + (lx - px) * k, ey = py + (ly - py) * k;
  c.beginPath();
  c.arc(px, py, 5, 0, Math.PI * 2);
  c.stroke();
  c.beginPath();
  c.moveTo(px, py);
  c.lineTo(ex, ey);
  const right = lx >= px;
  if (k > 0.98) c.lineTo(ex + (right ? 26 : -26), ey);
  c.stroke();
  c.font = font(F.mono(500), 21);
  c.letterSpacing = '1px';
  c.textBaseline = 'middle';
  c.textAlign = right ? 'left' : 'right';
  if (k > 0.98) typeOn(c, text, ex + (right ? 34 : -34), ey, t, t0 + 0.35, 0.4);
  c.restore();
}
