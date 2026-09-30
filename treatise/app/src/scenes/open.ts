// The title page: bone paper inside the crop-mark frame, the treatise's name in Cormorant italic, a
// frontispiece (Sevanavank, the day's destination: cut as an engraving, then developed into the
// photograph), the instrument at its prior, a footnote.
// The spark underlines the title, rests, then dives off the bottom of the page into the map (Табл. I).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, clearRT } from '../engine/gl';
import { LineBatch } from '../engine/lines';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { loadShot, loadPhoto, type Shot, type PhotoShot } from '../engine/footage';
import { clamp, ease, lerp, prog, pulse } from '../engine/util';
import { sparkHead, sparkParticles } from './_motifs';
import { Engraver, Photo, drawReadout, typeOn } from './_plate';

const WIN: [number, number, number, number] = [150, 700, 780, 600];

export default class Open extends Scene {
  eng = new Engraver();
  photo = new Photo();
  shot!: Shot;
  ph!: PhotoShot;
  L = new Layer2D();
  lines = new LineBatch(3000, { screen2D: true, blend: 'add' });

  override async init() {
    this.shot = await loadShot('vank_zoom');
    this.ph = await loadPhoto('vank_zoom');
  }

  /** The frontispiece plays at half speed and holds its last frame. */
  fl(lt: number) { return Math.min(lt * 0.5, 0.95); }

  override async prepare(lts: number[]) { await this.ph.prepare(lts.map((lt) => this.fl(lt))); }

  /** The spark: draws the underline (0.9–1.7), rests at its end, dives off the page (3.2–4.0). */
  head(lt: number) {
    const x0 = 150, x1 = 930, y = 560;
    if (lt < 0.9) return null;
    if (lt < 3.2) return { x: lerp(x0, x1, ease.inOutCubic(prog(lt, 0.9, 1.7))), y };
    const k = ease.inExpo(prog(lt, 3.2, 3.95));
    return { x: lerp(x1, 360, ease.outCubic(prog(lt, 3.2, 3.95))), y: lerp(y, 2150, k) };
  }

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const lt = f.lt, t = f.t;
    const exit = prog(lt, 3.45, 4.0);
    // the frontispiece: engraved in ink, cut from the top down by the burin; then the photograph develops
    clearRT(renderer, out, LIN.bone);
    const zoom = 1.02 + 0.03 * lt / 4, panY = 0.05;
    const develop = ease.inOutQuad(prog(lt, 1.5, 2.2));
    if (develop < 1) {
      this.eng.render(renderer, out, this.shot.at(this.fl(lt)), {
        paper: 1, win: WIN, zoom, panY, gain: 1.08, lift: 0.03, freq: 0.21, angle: 0.3,
        reveal: ease.outCubic(prog(lt, 0.5, 1.4)), revealSoft: 0.12,
      });
    }
    if (develop > 0) this.photo.render(renderer, out, this.ph.at(this.fl(lt)), { win: WIN, zoom, panY, develop: develop > 0.999 ? 1.01 : develop });

    const c = this.L.ctx;
    this.L.clear();
    c.save();
    c.textBaseline = 'alphabetic';
    // head: series line
    c.font = font(F.mono(500), 21);
    c.letterSpacing = '6px';
    c.fillStyle = rgba('ink', 0.7);
    typeOn(c, 'ТРАКТАТ № 1 · 27.IX.2026', 150, 250, lt, -0.2, 0.35);
    c.letterSpacing = '0px';
    // the title, rising from a clip, line by line on the beat (the first frame is the cover: already up)
    c.font = font(F.serif(600, true), 138);
    c.fillStyle = rgba('ink', 0.97);
    const lines = ['Один день', 'в Армении'];
    lines.forEach((s, i) => {
      const k = ease.outExpo(prog(lt, -0.3 + i * 0.5, 0.3 + i * 0.5));
      const y = 395 + i * 132;
      c.save();
      c.beginPath(); c.rect(100, y - 130, W, 162); c.clip();
      c.fillText(s, 146, y + (1 - k) * 150);
      c.restore();
    });
    // subtitle in the wide grotesk
    c.font = font(F.archivo(1250, 500), 25);
    c.letterSpacing = '5px';
    c.fillStyle = rgba('ink', 0.85);
    typeOn(c, 'ИЛЛЮСТРИРОВАННЫЙ ТРАКТАТ', 150, 612, lt, 1.0, 0.35);
    typeOn(c, 'В ДЕВЯТИ ТАБЛИЦАХ', 150, 648, lt, 1.2, 0.3);
    c.letterSpacing = '0px';
    // the frontispiece's frame and legend
    const fk = ease.outExpo(prog(lt, 0.5, 1.3));
    const [x, y, w, h] = WIN;
    c.strokeStyle = rgba('ink', 0.85);
    c.lineWidth = 1.4;
    c.beginPath();
    c.moveTo(x, y); c.lineTo(x + w * fk, y);
    c.moveTo(x + w, y + h); c.lineTo(x + w - w * fk, y + h);
    c.moveTo(x, y + h); c.lineTo(x, y + h - h * fk);
    c.moveTo(x + w, y); c.lineTo(x + w, y + h * fk);
    c.stroke();
    c.strokeStyle = rgba('ink', 0.4);
    c.strokeRect(x - 10, y - 10, w + 20, h + 20);
    c.font = font(F.serif(500, true), 30);
    c.fillStyle = rgba('ink', 0.8);
    typeOn(c, 'Фронтиспис. Вид, открывающийся исследователю к вечеру.', x, y + h + 56, lt, 1.6, 0.6);
    // the instrument at its prior
    drawReadout(c, 150, 1520, 0.02, { ink: true, scale: 1, alpha: clamp((lt - 1.9) * 5) });
    c.font = font(F.mono(400), 20);
    c.fillStyle = rgba('ink', 0.6);
    typeOn(c, 'априорная оценка', 150 + 270, 1520, lt, 2.1, 0.3);
    c.fillStyle = rgba('ink', 0.55);
    typeOn(c, '¹ Все измерения выполнены на глаз.', 150, 1690, lt, 2.5, 0.45);
    // the underline the spark leaves (ink hairline behind the glowing head)
    const hd = this.head(lt);
    const ux = lt < 0.9 ? 150 : lt < 3.2 ? hd!.x : 930;
    c.fillStyle = rgba('ink', 0.9);
    if (lt >= 0.9) c.fillRect(150, 559, ux - 150, 2);
    c.restore();
    comp.draw(renderer, this.L.upload(), out);

    const lb = this.lines;
    lb.clear();
    if (hd) {
      if (lt >= 3.2) {  // the dive drags a signal hairline off the page
        const pts = [] as { x: number; y: number }[];
        for (let i = 0; i <= 24; i++) { const h2 = this.head(lerp(3.2, lt, i / 24)); if (h2) pts.push(h2); }
        lb.polyline(pts, 2.4, [LIN.signal[0] * 1.3, LIN.signal[1] * 1.3, LIN.signal[2] * 1.3], 0.9);
      }
      sparkParticles(lb, t, (tb) => this.head(tb - f.start), { rate: 70, intensity: 0.8, life: 0.35, speed: 180 });
      sparkHead(lb, hd.x, hd.y, t, 0.9, 1);
    }
    lb.render(renderer, out);
    return {
      paper: 1, frame: 1 - ease.inExpo(exit), bloom: 0.45, vignette: 0.12, grain: 0.05,
      flash: 0.2 * pulse(lt, 0.5, 0.12), zoom: 1 + 0.01 * pulse(lt, 0.5, 0.15),
    };
  }
}
