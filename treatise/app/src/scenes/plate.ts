// A footage plate of the treatise: the day's clips engraved (see _plate.ts Engraver), cut on the beat
// inside the plate, with a diagram in the plate's own idiom drawn over them by the spark, a caption
// block (table number, Cormorant title, deadpan footnotes) and the P(восторг) instrument.
// One module serves every place; `params.id` picks the shots and the diagram.
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { LineBatch } from '../engine/lines';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { loadShot, type Shot } from '../engine/footage';
import { clamp, ease, lerp, prog, pulse, TAU, hash } from '../engine/util';
import { sparkHead, sparkParticles } from './_motifs';
import { Engraver, PVal, PLATE_MASK, drawCaption, drawReadout, callout, typeOn, setHalo } from './_plate';
import { LAKE } from './_lake';

interface Cut { shot: string; at: number; dur: number; zoom: [number, number]; pan?: [number, number, number, number]; rot?: number }
interface PlateDef {
  cuts: Cut[]; paper?: number; angle?: number; freq?: number; gain?: number; lift?: number; gamma?: number;
  table: string; title: string; notes: string[]; titleSize?: number;
}

const PLATES: Record<string, PlateDef> = {
  garni: {
    cuts: [
      { shot: 'garni_cols', at: 0, dur: 2, zoom: [1.02, 1.1] },
      { shot: 'garni_orange', at: 2, dur: 1, zoom: [1.06, 1.1] },
      { shot: 'garni_bath', at: 3, dur: 1, zoom: [1.0, 1.06] },
    ],
    table: 'ТАБЛ. II', title: 'Храм Гарни', notes: ['I в. н. э. · ионический ордер', 'колонн: 24 · богов в наличии: 0'],
  },
  basalt: {
    cuts: [
      { shot: 'basalt_arch', at: 0, dur: 1.5, zoom: [1.02, 1.1] },
      { shot: 'basalt_walk', at: 1.5, dur: 1.5, zoom: [1.0, 1.05] },
    ],
    angle: -0.25, table: 'ТАБЛ. III', title: 'Симфония камней', titleSize: 70,
    notes: ['базальт · шестигранный · остыл сам', 'объяснений не даёт'],
  },
  geghard: {
    cuts: [
      { shot: 'geghard_cliff', at: 0, dur: 2, zoom: [1.0, 1.07] },
      { shot: 'geghard_yard', at: 2, dur: 1, zoom: [1.04, 1.08] },
      { shot: 'geghard_tunnel', at: 3, dur: 1, zoom: [1.0, 1.1] },
    ],
    paper: 1, gain: 1.05, lift: 0.02, table: 'ТАБЛ. IV', title: 'Гегард',
    notes: ['монастырь, частично высечен в скале', 'IV–XIII вв. · скала − скала = храм'],
  },
  sevan: {
    cuts: [
      { shot: 'sevan_beach', at: 0, dur: 2, zoom: [1.0, 1.06] },
      { shot: 'sevan_sun', at: 2, dur: 2, zoom: [1.04, 1.1] },
    ],
    angle: 0.05, table: 'ТАБЛ. VI', title: 'Озеро Севан', notes: ['1900 м над уровнем моря', 't° воды: холодно¹ · ¹ проверено лично'],
  },
  jetski: {
    cuts: [
      { shot: 'jet_tilt', at: 0, dur: 1.5, zoom: [1.05, 1.12] },
      { shot: 'jet_spray', at: 1.5, dur: 2.5, zoom: [1.04, 1.1] },
    ],
    angle: 0.1, table: 'ТАБЛ. VII', title: 'Гидроцикл', notes: ['амплитуда: да', 'частота: весело'],
  },
  vank: {
    cuts: [
      { shot: 'vank_church', at: 0, dur: 2, zoom: [1.0, 1.06] },
      { shot: 'vank_zoom', at: 2, dur: 1, zoom: [1.02, 1.08] },
      { shot: 'vank_lake', at: 3, dur: 1, zoom: [1.0, 1.05] },
    ],
    paper: 1, gain: 1.05, lift: 0.02, table: 'ТАБЛ. VIII', title: 'Севанаванк',
    notes: ['IX в. · крестово-купольный', 'полуостров, до XX в. — остров'],
  },
  peninsula: {
    cuts: [
      { shot: 'pen_clouds', at: 0, dur: 2, zoom: [1.0, 1.05] },
      { shot: 'pen_ridge', at: 2, dur: 2, zoom: [1.02, 1.07] },
    ],
    angle: 0.12, table: 'ТАБЛ. IX', title: 'Вид с полуострова', titleSize: 66,
    notes: ['16:22 · облака стекают с хребта', 'тип облака: не классифицирован'],
  },
};

type P2 = { x: number; y: number };

export default class Plate extends Scene {
  def!: PlateDef;
  pid = '';
  shots = new Map<string, Shot>();
  eng = new Engraver();
  L = new Layer2D();
  lines = new LineBatch(8000, { screen2D: true, blend: 'add' });
  pv = new PVal();

  override async init() {
    this.pid = String(this.ctx.params.id);
    this.def = PLATES[this.pid]!;
    for (const c of this.def.cuts) if (!this.shots.has(c.shot)) this.shots.set(c.shot, await loadShot(c.shot));
  }

  cutAt(lt: number) {
    const cs = this.def.cuts;
    let c = cs[0]!;
    for (const x of cs) if (lt >= x.at) c = x;
    return c;
  }

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const d = this.def, lt = f.lt, t = f.t, paper = d.paper ?? 0;
    const cut = this.cutAt(lt);
    const cl = lt - cut.at, cp = clamp(cl / cut.dur);
    const shot = this.shots.get(cut.shot)!;
    // each cut lands with a small punch-in that settles, then drifts on its own zoom
    const punch = 0.035 * Math.pow(0.5, cl / 0.12);
    const zoom = lerp(cut.zoom[0], cut.zoom[1], ease.inOutQuad(cp)) + punch;
    this.eng.render(renderer, out, shot.at(cl), {
      zoom, rot: cut.rot ?? 0, paper, angle: d.angle ?? 0.35, freq: d.freq, gain: d.gain ?? 1.05, lift: d.lift ?? 0.05,
      gamma: d.gamma ?? (paper > 0.5 ? 1.25 : 1.35), mask: PLATE_MASK,
      reveal: cut.at === 0 ? ease.outCubic(clamp(lt / 0.45)) : 1,
    });

    // ---- overlay
    const c = this.L.ctx;
    this.L.clear();
    const lb = this.lines;
    lb.clear();
    const ink = paper > 0.5;
    // a soft halo of the ground colour keeps hairlines and labels legible over the engraving
    c.shadowColor = rgba(ink ? 'bone' : 'ink', 0.95);
    c.shadowBlur = 9;
    setHalo(rgba(ink ? 'bone' : 'ink', 0.85));
    const head = this.diagram(c, lb, f, cut, cl, shot, zoom);
    setHalo(null);
    // the bottom block (clear of the stories UI): caption on the left, the instrument on the right
    drawCaption(c, lt, 0.3, { table: d.table, title: d.title, notes: d.notes, ink, size: d.titleSize ?? 74 });
    drawReadout(c, 790, 1500, this.pv.value(t), { ink, flash: this.pv.flash(t), scale: 0.9 });
    comp.draw(renderer, this.L.upload(), out);
    if (head) {
      sparkParticles(lb, t, () => head, { rate: 70, intensity: 0.8, life: 0.35, speed: 200 });
      sparkHead(lb, head.x, head.y, t, 0.8, 1);
    }
    lb.render(renderer, out);
    const cutHit = pulse(lt, cut.at, 0.08);
    return {
      paper, bloom: 0.6, flash: cut.at > 0 ? 0.09 * cutHit : 0.3 * pulse(lt, 0, 0.1),
      shake: [0, 0], zoom: 1 + 0.01 * pulse(lt, cut.at, 0.12), vignette: paper ? 0.15 : 0.35,
    };
  }

  /** The plate's own diagram. Returns the spark's head position (or null when it rests). */
  diagram(c: CanvasRenderingContext2D, lb: LineBatch, f: Frame, cut: Cut, cl: number, shot: Shot, zoom: number): P2 | null {
    const lt = f.lt, t = f.t, ink = (this.def.paper ?? 0) > 0.5;
    const fg = ink ? 'ink' : 'bone';
    const toScreen = (u: number, v: number) => ({ x: (u - 0.5) * zoom * W + W / 2, y: (v - 0.5) * zoom * H + H / 2 });
    c.save();
    c.lineWidth = 1.3;
    let head: P2 | null = null;
    switch (this.pid) {
      case 'garni': {
        // detail A: an Ionic volute, drawn as a logarithmic spiral by the spark
        const cx = 300, cy = 540, R = 120, p = ease.inOutCubic(prog(lt, 0.4, 1.9));
        c.strokeStyle = rgba(fg, 0.85);
        c.beginPath();
        const n = Math.floor(240 * p);
        let last: P2 = { x: cx, y: cy };
        for (let i = 0; i <= n; i++) {
          const a = (i / 240) * TAU * 2.6, r = R * Math.exp(-0.26 * a);
          const x = cx + r * Math.cos(a + 2.2), y = cy + r * Math.sin(a + 2.2);
          if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
          last = { x, y };
        }
        c.stroke();
        if (p > 0 && p < 1) head = last;
        c.strokeStyle = rgba(fg, 0.35);
        c.strokeRect(cx - R - 26, cy - R - 26, 2 * R + 52, 2 * R + 52);
        c.font = font(F.mono(500), 20);
        c.fillStyle = rgba(fg, 0.75);
        typeOn(c, 'ДЕТАЛЬ А · ВОЛЮТА', cx - R - 26, cy + R + 60, lt, 1.0, 0.4);
        // the only orange thing in the frame gets a label
        if (cut.shot === 'garni_orange') {
          const s = shot.signalAt(cl);
          if (s && s[2] > 0.005) {
            const p0 = toScreen(s[0], s[1]);
            callout(c, p0.x, p0.y, p0.x < W / 2 ? p0.x + 150 : p0.x - 150, Math.max(420, p0.y - 230), 'ОРАНЖЕВЫЙ ОБЪЕКТ · 1 ШТ.', lt, cut.at + 0.05, { signal: true });
          }
        }
        break;
      }
      case 'basalt': {
        // hexagonal lattice over the columns; the spark traces one cell per beat
        const r = 58, h = r * Math.sqrt(3);
        const ox = 140, oy = 700;
        c.strokeStyle = rgba(fg, 0.22);
        const hex = (x: number, y: number) => { c.beginPath(); for (let k = 0; k <= 6; k++) { const a = TAU * k / 6; k ? c.lineTo(x + r * Math.cos(a), y + r * Math.sin(a)) : c.moveTo(x + r, y); } c.stroke(); };
        const cells: P2[] = [];
        for (let j = 0; j < 5; j++) for (let i = 0; i < 6; i++) {
          const x = ox + i * r * 1.5, y = oy + j * h + (i % 2 ? h / 2 : 0);
          cells.push({ x, y });
          if (hash(i, j, 3) < prog(lt, 0.2, 1.4)) hex(x, y);
        }
        const beat = Math.floor(lt / 0.5), bp = (lt / 0.5) % 1;
        const cell = cells[(beat * 7 + 3) % cells.length]!;
        c.strokeStyle = rgba('signal', 0.9);
        c.lineWidth = 2;
        const q = ease.inOutCubic(clamp(bp / 0.8));
        c.beginPath();
        const segs = 6 * q;
        for (let k = 0; k <= Math.ceil(segs); k++) {
          const a = TAU * Math.min(k, segs) / 6;
          const x = cell.x + r * Math.cos(a), y = cell.y + r * Math.sin(a);
          k ? c.lineTo(x, y) : c.moveTo(x, y);
          if (k === Math.ceil(segs)) head = { x, y };
        }
        c.stroke();
        c.font = font(F.mono(500), 21);
        c.fillStyle = rgba(fg, 0.8);
        typeOn(c, `ГРАНЕЙ: 6 · СТОЛБОВ: ${Math.floor(lerp(1, 9999, ease.outExpo(prog(lt, 0.3, 2.4))))}`, 72, 560, lt, 0.2, 0.3);
        break;
      }
      case 'geghard': {
        // section A–A: rock (hatched) minus a church = the monastery
        const bx = 610, by = 470, bw = 390, bh = 300, p = ease.inOutCubic(prog(lt, 0.5, 2.2));
        // on a card of bare paper, cut down from the top
        const card = ease.outExpo(prog(lt, 0.3, 0.8));
        c.save();
        c.fillStyle = rgba('bone', 1);
        c.fillRect(bx - 24, by - 24, bw + 48, (bh + 92) * card);
        c.beginPath(); c.rect(bx - 24, by - 24, bw + 48, (bh + 92) * card); c.clip();
        c.shadowBlur = 0;
        c.save();
        c.beginPath(); c.rect(bx, by, bw, bh); c.clip();
        c.strokeStyle = rgba('ink', 0.55);
        for (let k = -bh; k < bw; k += 11) { c.beginPath(); c.moveTo(bx + k, by + bh); c.lineTo(bx + k + bh, by); c.stroke(); }
        c.restore();
        c.strokeStyle = rgba('ink', 0.9);
        c.strokeRect(bx, by, bw, bh);
        // the void: nave, apse and a drum with its cone, cut out of the rock
        const vx = bx + bw / 2, vy = by + bh - 30;
        const pts: P2[] = [
          { x: vx - 110, y: vy }, { x: vx - 110, y: vy - 120 }, { x: vx - 40, y: vy - 120 }, { x: vx - 40, y: vy - 175 },
          { x: vx, y: vy - 225 }, { x: vx + 40, y: vy - 175 }, { x: vx + 40, y: vy - 120 }, { x: vx + 110, y: vy - 120 }, { x: vx + 110, y: vy },
        ];
        c.fillStyle = rgba('bone', 1);
        c.beginPath(); pts.forEach((q, i) => (i ? c.lineTo(q.x, q.y) : c.moveTo(q.x, q.y))); c.closePath();
        c.save(); c.globalAlpha = p; c.fill(); c.restore();
        const L = pts.length - 1, sP = p * L, k = Math.floor(sP);
        c.strokeStyle = rgba('signal', 0.95);
        c.lineWidth = 2.2;
        c.beginPath();
        for (let i = 0; i <= Math.min(k, L); i++) (i ? c.lineTo(pts[i]!.x, pts[i]!.y) : c.moveTo(pts[i]!.x, pts[i]!.y));
        if (k < L) { const a = pts[k]!, b = pts[k + 1]!, u = sP - k; head = { x: lerp(a.x, b.x, u), y: lerp(a.y, b.y, u) }; c.lineTo(head.x, head.y); }
        c.stroke();
        c.font = font(F.mono(500), 17);
        c.fillStyle = rgba('ink', 0.8);
        typeOn(c, 'РАЗРЕЗ А–А · МЕТОД: ВЫЧИТАНИЕ', bx, by + bh + 42, lt, 0.8, 0.5);
        c.restore();
        break;
      }
      case 'sevan': {
        // lake inset: the shoreline as a contour, the altitude as a spot height
        const ox = 660, oy = 400, s = 330;
        const p = ease.inOutCubic(prog(lt, 0.3, 1.8));
        // on a card of bare ground, cut down from the top
        const card = ease.outExpo(prog(lt, 0.1, 0.6));
        c.save();
        c.shadowBlur = 0;
        c.fillStyle = rgba('ink', 1);
        c.fillRect(ox - 30, oy - 30, s * 0.9 + 60, (s + 130) * card);
        c.strokeStyle = rgba(fg, 0.5);
        c.strokeRect(ox - 30, oy - 30, s * 0.9 + 60, (s + 130) * card);
        c.beginPath(); c.rect(ox - 30, oy - 30, s * 0.9 + 60, (s + 130) * card); c.clip();
        // the lake in water lines inside its shore
        c.save();
        c.beginPath();
        LAKE.forEach((q, i) => (i ? c.lineTo(ox + q[0] * s, oy + q[1] * s) : c.moveTo(ox + q[0] * s, oy + q[1] * s)));
        c.clip();
        c.fillStyle = rgba(fg, 0.3 * clamp((lt - 1.2) * 3));
        for (let yy = oy; yy < oy + s; yy += 7) c.fillRect(ox, yy, s, 1);
        c.restore();
        const shore = LAKE;
        const n = Math.floor(shore.length * p);
        c.strokeStyle = rgba(fg, 0.9);
        c.beginPath();
        for (let i = 0; i < n; i++) { const q = shore[i]!; const x = ox + q[0] * s, y = oy + q[1] * s; i ? c.lineTo(x, y) : c.moveTo(x, y); if (i === n - 1 && p < 1) head = { x, y }; }
        c.stroke();
        c.font = font(F.mono(500), 20);
        c.fillStyle = rgba(fg, 0.8);
        typeOn(c, '▲ 1900 м', ox + 40, oy + s * 0.42, lt, 1.6, 0.3);
        typeOn(c, 'ПЛОЩАДЬ ≈ 1260 км²', ox - 10, oy + s + 60, lt, 1.8, 0.4);
        c.restore();
        // sun glints get flagged when they appear
        if (cut.shot === 'sevan_sun') {
          const sg = shot.signalAt(cl);
          const p0 = sg ? toScreen(sg[0], sg[1]) : null;
          // only when the glitter sits inside the picture (not up in the sky by the sun)
          if (sg && p0 && sg[2] > 0.0003 && p0.y > 520 && p0.y < 1150 && p0.x > 120 && p0.x < 960) {
            callout(c, p0.x, p0.y, p0.x < W / 2 ? p0.x + 160 : p0.x - 160, Math.min(1150, p0.y + 200), 'БЛИКИ · ≈10⁴ ШТ.', lt, cut.at + 0.1, { signal: true });
          }
        }
        break;
      }
      case 'jetski': {
        // oscilloscope graticule; the trace is the wake, its amplitude grows into the spray
        const gx = 72, gy = 430, gw = W - 144, gh = 560, cols = 10, rows = 8;
        c.strokeStyle = rgba(fg, 0.3);
        for (let i = 0; i <= cols; i++) { c.beginPath(); c.moveTo(gx + gw * i / cols, gy); c.lineTo(gx + gw * i / cols, gy + gh); c.stroke(); }
        for (let j = 0; j <= rows; j++) { c.beginPath(); c.moveTo(gx, gy + gh * j / rows); c.lineTo(gx + gw, gy + gh * j / rows); c.stroke(); }
        c.strokeStyle = rgba(fg, 0.6);
        c.beginPath(); c.moveTo(gx, gy + gh / 2); c.lineTo(gx + gw, gy + gh / 2); c.moveTo(gx + gw / 2, gy); c.lineTo(gx + gw / 2, gy + gh); c.stroke();
        const amp = lerp(0.12, 0.85, ease.inCubic(prog(lt, 0.3, 2.3))) * (lt > 3.5 ? 0.6 : 1);
        const sweep = clamp(0.08 + (lt % 1.0) / 0.6);
        const pts: P2[] = [];
        for (let i = 0; i <= 200 * sweep; i++) {
          const u = i / 200, x = gx + u * gw;
          const y = gy + gh / 2 - amp * gh * 0.42 * (Math.sin(u * 23 + t * 9) * 0.6 + Math.sin(u * 61 - t * 13) * 0.3 * (0.5 + f.a.high) + (hash(i, Math.floor(t * 10)) - 0.5) * 0.25 * amp);
          pts.push({ x, y });
        }
        if (pts.length > 1) {
          for (let i = 1; i < pts.length; i++) lb.seg2(pts[i - 1]!.x, pts[i - 1]!.y, pts[i]!.x, pts[i]!.y, 3, [LIN.signal[0] * 2, LIN.signal[1] * 2, LIN.signal[2] * 2], 0.9);
          head = pts[pts.length - 1]!;
        }
        c.font = font(F.mono(500), 20);
        c.fillStyle = rgba(fg, 0.8);
        typeOn(c, 'CH1 · БРЫЗГИ · 2 м/дел', gx, gy - 18, lt, 0.2, 0.4);
        c.textAlign = 'right';
        typeOn(c, 'TRIG ↑ · АВТО', gx + gw, gy - 18, lt, 0.4, 0.3);
        break;
      }
      case 'vank': {
        // plan of a cross-domed church, drawn in ink on the paper plate
        const cx = 290, cy = 580, a = 50, p = ease.inOutCubic(prog(lt, 0.4, 2.3));
        const card = ease.outExpo(prog(lt, 0.2, 0.7));
        c.save();
        c.fillStyle = rgba('bone', 1);
        c.fillRect(cx - 3 * a - 40, cy - 290, 6 * a + 80, 420 * card);
        c.strokeStyle = rgba('ink', 0.6);
        c.lineWidth = 1.2;
        c.strokeRect(cx - 3 * a - 40, cy - 290, 6 * a + 80, 420 * card);
        c.beginPath(); c.rect(cx - 3 * a - 40, cy - 290, 6 * a + 80, 420 * card); c.clip();
        c.shadowBlur = 0;
        const plan: P2[] = [
          [-1, -3], [1, -3], [1, -1], [3, -1], [3, 1], [1, 1], [1, 3.2], [0.6, 3.7], [0, 3.85], [-0.6, 3.7], [-1, 3.2], [-1, 1], [-3, 1], [-3, -1], [-1, -1], [-1, -3],
        ].map(([x, y]) => ({ x: cx + x! * a, y: cy - y! * a * 0.62 - 20 }));
        const L = plan.length - 1, sP = p * L, k = Math.floor(sP);
        c.strokeStyle = rgba('ink', 0.9);
        c.lineWidth = 2;
        c.beginPath();
        for (let i = 0; i <= Math.min(k, L); i++) (i ? c.lineTo(plan[i]!.x, plan[i]!.y) : c.moveTo(plan[i]!.x, plan[i]!.y));
        if (k < L) { const A = plan[k]!, B = plan[k + 1]!, u = sP - k; head = { x: lerp(A.x, B.x, u), y: lerp(A.y, B.y, u) }; c.lineTo(head.x, head.y); }
        c.stroke();
        c.lineWidth = 1.2;
        c.strokeStyle = rgba('ink', 0.55 * clamp((lt - 2.3) * 3));
        c.beginPath(); c.arc(cx, cy - 20, a * 0.8, 0, TAU); c.stroke();          // the drum
        c.beginPath(); c.moveTo(cx - 3 * a, cy + a * 1.2); c.lineTo(cx + 3 * a, cy + a * 1.2); c.stroke();  // dimension
        c.font = font(F.mono(500), 19);
        c.fillStyle = rgba('ink', 0.8 * clamp((lt - 2.3) * 3));
        c.textAlign = 'center';
        c.fillText('≈ 12 м', cx, cy + a * 1.2 + 26);
        c.textAlign = 'left';
        c.fillStyle = rgba('ink', 0.8);
        c.font = font(F.mono(500), 17);
        typeOn(c, 'ПЛАН · КРЕСТ В КВАДРАТЕ', cx - 3 * a - 16, cy - 250, lt, 0.4, 0.4);
        c.restore();
        break;
      }
      case 'peninsula': {
        // a levelling line across the horizon, with ticks; the clouds pour over the ridge above it
        const y = lt < 2 ? 700 : 820, p = ease.outExpo(prog(lt % 2, 0.15, 1.0));
        c.strokeStyle = rgba(fg, 0.75);
        c.beginPath(); c.moveTo(72, y); c.lineTo(72 + (W - 144) * p, y); c.stroke();
        for (let i = 0; i <= 12 * p; i++) { const x = 72 + (W - 144) * i / 12; c.beginPath(); c.moveTo(x, y - (i % 6 ? 8 : 16)); c.lineTo(x, y); c.stroke(); }
        head = p < 1 ? { x: 72 + (W - 144) * p, y } : null;
        c.font = font(F.mono(500), 20);
        c.fillStyle = rgba(fg, 0.8);
        typeOn(c, lt < 2 ? 'ГОРИЗОНТ · ВЫРОВНЕН' : 'ХРЕБЕТ · ОБЛАКА ПЕРЕЛИВАЮТСЯ ЧЕРЕЗ КРАЙ', 72, y - 30, lt % 2, 0.3, 0.5);
        break;
      }
    }
    c.restore();
    return head;
  }
}
