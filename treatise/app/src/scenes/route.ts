// The route plates (Табл. I and V): the contour-engraved relief between Yerevan and Lake Sevan, the
// day's road as a faint dashed plan, and the spark dragging the travelled leg in signal orange while
// an odometer and a clock run. `params.leg`: 1 Yerevan→Garni, 2 Garni→Geghard, 3 Geghard→Sevan (the
// riser: the camera pulls back over the whole route, then dives into the lake on the drop).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, H } from '../engine/gl';
import { LineBatch } from '../engine/lines';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { loadPhoto, type PhotoShot } from '../engine/footage';
import { clamp, ease, lerp, prog, pulse, polylineLengths, pointAtLength, type V2 } from '../engine/util';
import { sparkHead, sparkParticles } from './_motifs';
import { Photo, TerrainMap, loadMap, mapToScreen, type MapCam, type MapMeta, PVal, drawCaption, drawReadout, typeOn, setHalo } from './_plate';

// the road, as (lat, lon) waypoints
const LL: Record<number, [number, number][]> = {
  1: [[40.1776, 44.5126], [40.170, 44.560], [40.172, 44.600], [40.180, 44.640], [40.160, 44.690], [40.128, 44.715], [40.1125, 44.7302]],
  2: [[40.1125, 44.7302], [40.125, 44.760], [40.135, 44.790], [40.1403, 44.8183]],
  3: [[40.1403, 44.8183], [40.135, 44.790], [40.125, 44.760], [40.1125, 44.7302], [40.160, 44.690], [40.200, 44.640], [40.270, 44.628],
      [40.330, 44.660], [40.400, 44.700], [40.497, 44.766], [40.530, 44.830], [40.545, 44.900], [40.550, 44.960], [40.558, 45.004]],
};
const toMap = (m: MapMeta, [lat, lon]: [number, number]): V2 => ({
  x: (lon - m.bbox[0]) / (m.bbox[2] - m.bbox[0]), y: (m.bbox[3] - lat) / (m.bbox[3] - m.bbox[1]),
});

interface Leg {
  km: [number, number]; clock: [number, number]; spark: [number, number];
  cam: (lt: number, m: MapMeta) => MapCam;
  inset?: { shot: string; win: [number, number, number, number]; t: [number, number]; label: string };
  cap?: { table: string; title: string; notes: string[] };
}
const hm = (h: number, m: number) => h * 60 + m;
const LEGS: Record<number, Leg> = {
  1: {
    km: [0, 28], clock: [hm(10, 29), hm(10, 56)], spark: [0.15, 1.75],
    cam: (lt) => ({ cx: lerp(0.22, 0.28, ease.outCubic(clamp(lt / 2))), cy: 0.66, scale: lerp(3500, 3050, ease.outExpo(clamp(lt / 1.2))), rot: 0 }),
    inset: { shot: 'road1', win: [640, 390, 360, 640], t: [0.05, 2], label: 'РИС. 1а · ДОРОГА · ×2' },
    cap: { table: 'ТАБЛ. I', title: 'Маршрут', notes: ['Ереван → Гарни · 28 км', 'выезд 10:29 · кофе: взят'] },
  },
  2: {
    km: [28, 38], clock: [hm(12, 38), hm(12, 52)], spark: [0.05, 0.8],
    cam: (lt) => ({ cx: 0.395, cy: 0.72, scale: lerp(7200, 6500, ease.outExpo(clamp(lt / 0.6))), rot: 0 }),
  },
  3: {
    km: [38, 113], clock: [hm(13, 5), hm(14, 50)], spark: [0.3, 3.4],
    cam: (lt, m) => {
      const g = toMap(m, [40.1403, 44.8183]), s = toMap(m, [40.558, 45.004]);
      const out = ease.outExpo(clamp(lt / 0.9));
      const dive = ease.inExpo(prog(lt, 3.35, 4.0));
      const cx = lerp(lerp(g.x, 0.43, out), s.x, dive), cy = lerp(lerp(g.y, 0.604, out), s.y, dive);
      const wide = lerp(1700, 1820, prog(lt, 0.9, 3.35));
      return { cx, cy, scale: Math.exp(lerp(lerp(Math.log(5200), Math.log(wide), out), Math.log(16000), dive)), rot: 0 };
    },
    inset: { shot: 'road2', win: [690, 700, 324, 576], t: [0.8, 3.3], label: 'РИС. 5а · М-4 · УСКОРЕНО ×3' },
    cap: { table: 'ТАБЛ. V', title: 'Перегон', notes: ['Гегард → Севан · 75 км', 'скорость: разрешённая¹'] },
  },
};

const PLACES: { key: string; name: string; time: string; leg: number }[] = [
  { key: 'yerevan', name: 'ЕРЕВАН', time: '10:29', leg: 0 },
  { key: 'garni', name: 'ГАРНИ', time: '10:56', leg: 1 },
  { key: 'geghard', name: 'ГЕГАРД', time: '12:52', leg: 2 },
  { key: 'sevan', name: 'СЕВАН', time: '14:50', leg: 3 },
];

export default class Route extends Scene {
  leg = 1;
  L!: Leg;
  map!: { meta: MapMeta; tex: THREE.DataTexture };
  terrain = new TerrainMap();
  photo = new Photo();
  shot: PhotoShot | null = null;
  layer = new Layer2D();
  lines = new LineBatch(6000, { screen2D: true, blend: 'add' });
  pv = new PVal();
  paths: { pts: V2[]; len: Float32Array }[] = [];

  override async init() {
    this.leg = Number(this.ctx.params.leg ?? 1);
    this.L = LEGS[this.leg]!;
    this.map = await loadMap();
    for (let k = 1; k <= 3; k++) {
      const pts = LL[k]!.map((p) => toMap(this.map.meta, p));
      this.paths[k] = { pts, len: polylineLengths(pts) };
    }
    if (this.L.inset) this.shot = await loadPhoto(this.L.inset.shot);
  }

  override async prepare(lts: number[]) {
    const ins = this.L.inset;
    if (ins && this.shot) await this.shot.prepare(lts.map((lt) => Math.max(0, lt - ins.t[0])));
  }

  /** Map position of the spark on this leg at local time lt, and the leg's 0..1 progress. */
  head(lt: number) {
    const [a, b] = this.L.spark;
    const p = this.leg === 3 ? ease.inOutQuad(prog(lt, a, b)) : ease.inOutCubic(prog(lt, a, b));
    const P = this.paths[this.leg]!;
    const q = pointAtLength(P.pts, P.len, p * P.len[P.len.length - 1]!);
    return { m: { x: q.x, y: q.y }, p };
  }

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const { meta, tex } = this.map;
    const lt = f.lt, t = f.t, L = this.L;
    const cam = L.cam(lt, meta);
    const dive = this.leg === 3 ? prog(lt, 3.35, 4.0) : 0;
    this.terrain.render(renderer, out, tex, cam, { lit: 1, mask: [-1, -0.9, 0.68, 0.78] });

    // the inset: footage of the road in its own window, cut in from the top
    const ins = L.inset;
    let insetK = 0;
    if (ins && this.shot) {
      insetK = clamp((lt - ins.t[0]) / 0.35) * (1 - clamp((lt - ins.t[1]) / 0.2));
      if (insetK > 0) {
        this.photo.render(renderer, out, this.shot.at(Math.max(0, lt - ins.t[0])), {
          win: ins.win, zoom: 1.05, develop: ease.outCubic(clamp((lt - ins.t[0]) / 0.4)) * 1.01, fade: 1 - clamp(insetK * 1.5),
        });
      }
    }

    const c = this.layer.ctx;
    this.layer.clear();
    const lb = this.lines;
    lb.clear();
    const S = (m: V2) => mapToScreen(cam, m.x, m.y);
    c.shadowColor = rgba('ink', 0.95);
    c.shadowBlur = 9;
    setHalo(rgba('ink', 0.85));

    // the plan: the whole day's road, dashed, faint; legs already driven, solid bone
    c.save();
    c.lineCap = 'round';
    for (let k = 1; k <= 3; k++) {
      const pts = this.paths[k]!.pts.map(S);
      c.beginPath();
      pts.forEach((p, i) => (i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)));
      if (k < this.leg) { c.setLineDash([]); c.strokeStyle = rgba('bone', 0.75); c.lineWidth = 2; }
      else { c.setLineDash([2, 9]); c.strokeStyle = rgba('bone', 0.45); c.lineWidth = 1.6; }
      c.stroke();
    }
    c.restore();

    // the travelled part of this leg: signal line (glowing), the spark at its end
    const { m: hmap, p } = this.head(lt);
    const P = this.paths[this.leg]!;
    const total = P.len[P.len.length - 1]!;
    const trail: V2[] = [];
    for (let i = 0; i < P.pts.length && P.len[i]! <= p * total; i++) trail.push(S(P.pts[i]!));
    const hs = S(hmap);
    trail.push(hs);
    lb.polyline(trail, 3.2, [LIN.signal[0] * 1.6, LIN.signal[1] * 1.6, LIN.signal[2] * 1.6], 0.95);

    // places
    const meta2 = meta.places;
    for (const pl of PLACES) {
      if (pl.leg > this.leg) continue;
      const arrived = pl.leg < this.leg || p >= 0.999;
      const at = pl.leg < this.leg ? -10 : L.spark[1];
      const pos = S({ x: meta2[pl.key]![0], y: meta2[pl.key]![1] });
      if (pos.x < -200 || pos.x > W + 200 || pos.y < -200 || pos.y > H + 200) continue;
      const k = pl.leg === this.leg ? ease.outExpo(clamp((lt - at) / 0.4)) : 1;
      if (pl.leg === this.leg && !arrived) continue;
      const fg = pl.leg === this.leg ? 'signal' : 'bone';
      c.strokeStyle = rgba(fg, 0.9);
      c.fillStyle = rgba(fg, 0.95);
      c.lineWidth = 1.5;
      c.beginPath(); c.arc(pos.x, pos.y, 8, 0, Math.PI * 2); c.stroke();
      c.beginPath(); c.arc(pos.x, pos.y, 2.6, 0, Math.PI * 2); c.fill();
      if (pl.leg === this.leg) {  // arrival ring
        const r = 8 + 60 * ease.outExpo(clamp((lt - at) / 0.6));
        c.strokeStyle = rgba('signal', 0.8 * (1 - clamp((lt - at) / 0.6)));
        c.beginPath(); c.arc(pos.x, pos.y, r, 0, Math.PI * 2); c.stroke();
      }
      c.font = font(F.mono(500), 22);
      c.letterSpacing = '4px';
      c.fillStyle = rgba(fg === 'signal' ? 'signal' : 'bone', 0.95);
      const right = pl.key !== 'yerevan' || pos.x < 200;
      c.textAlign = right ? 'left' : 'right';
      c.save();
      c.beginPath(); c.rect(right ? pos.x + 18 : pos.x - 400, pos.y - 40, 382 * k, 70); c.clip();
      c.fillText(pl.name, right ? pos.x + 20 : pos.x - 20, pos.y - 4);
      c.font = font(F.mono(400), 18);
      c.letterSpacing = '2px';
      c.fillStyle = rgba('bone', 0.6);
      c.fillText(pl.time, right ? pos.x + 20 : pos.x - 20, pos.y + 22);
      c.restore();
      c.letterSpacing = '0px';
      c.textAlign = 'left';
    }

    // inset frame + label
    if (ins && insetK > 0) {
      const [x, y, w, h] = ins.win;
      c.save();
      c.globalAlpha = insetK;
      c.strokeStyle = rgba('bone', 0.8);
      c.lineWidth = 1.3;
      c.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
      c.font = font(F.mono(500), 18);
      c.letterSpacing = '2px';
      c.fillStyle = rgba('bone', 0.75);
      typeOn(c, ins.label, x, y + h + 30, lt, ins.t[0] + 0.2, 0.4);
      c.restore();
    }

    // instruments: odometer + clock (top), P(восторг)
    const kmNow = lerp(L.km[0], L.km[1], p);
    const clk = Math.floor(lerp(L.clock[0], L.clock[1], p));
    c.save();
    c.font = font(F.mono(500), 17);
    c.letterSpacing = '3px';
    c.fillStyle = rgba('bone', 0.62);
    c.fillText('ПРОЙДЕНО', 72, 262);
    c.fillText('ВРЕМЯ', 400, 262);
    c.letterSpacing = '0px';
    c.font = font(F.mono(400), 52);
    c.fillStyle = rgba('bone', 0.94);
    c.fillText(`${kmNow.toFixed(1).padStart(5, '0')} км`, 70, 312);
    c.fillText(`${String(Math.floor(clk / 60)).padStart(2, '0')}:${String(clk % 60).padStart(2, '0')}`, 398, 312);
    c.restore();
    drawReadout(c, 790, 1500, this.pv.value(t), { flash: this.pv.flash(t), scale: 0.9 });
    if (L.cap) drawCaption(c, lt, 0.15, { ...L.cap, size: 74 });
    if (this.leg === 3) {
      c.font = font(F.mono(400), 20);
      c.fillStyle = rgba('bone', 0.55 * clamp((lt - 1.2) * 3));
      c.fillText('¹ по нашим данным', 72, 1660);
    }
    setHalo(null);
    comp.draw(renderer, this.layer.upload(), out);

    // the spark, with its sputter (particles are born along the head's screen path)
    if (p > 0 && p < 1) {
      sparkParticles(lb, t, (tb) => {
        const l = tb - f.start;
        const h = this.head(l);
        if (h.p <= 0 || h.p >= 1) return null;
        return mapToScreen(L.cam(l, meta), h.m.x, h.m.y);
      }, { rate: 80, intensity: 0.9, life: 0.4, speed: 220 });
      sparkHead(lb, hs.x, hs.y, t, 1, 1.1);
    }
    lb.render(renderer, out);

    return {
      paper: 0, bloom: 0.6, vignette: 0.4,
      flash: 0.35 * pulse(lt, L.spark[1], 0.1) + 1.2 * dive * dive,
      zoom: 1 + 0.012 * pulse(lt, L.spark[1], 0.15),
    };
  }
}
