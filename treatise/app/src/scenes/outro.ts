// The conclusion (Табл. X): the last walk along the peninsula, engraved; the instrument reaches 1.00
// and then breaks its own cap; the result is set as a numbered equation; the crop-mark frame closes
// in and the engraving is taken back to ink while the music fades.
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W } from '../engine/gl';
import { LineBatch } from '../engine/lines';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { loadShot, type Shot } from '../engine/footage';
import { clamp, ease, lerp, prog, pulse } from '../engine/util';
import { sparkHead, sparkParticles } from './_motifs';
import { Engraver, drawCaption, drawReadout, typeOn } from './_plate';

export default class Outro extends Scene {
  eng = new Engraver();
  shot!: Shot;
  L = new Layer2D();
  lines = new LineBatch(3000, { screen2D: true, blend: 'add' });

  override async init() { this.shot = await loadShot('pen_walk'); }

  /** The value: 1.00 from the start, over the cap from 1.5 s. */
  value(lt: number) { return 1 + 0.03 * ease.outExpo(prog(lt, 1.5, 2.3)) + 0.004 * Math.sin(lt * 7) * prog(lt, 2.3, 2.6); }

  /** The spark underlines the equation (3.1–3.7), then rests at its end. */
  head(lt: number) {
    if (lt < 3.1) return null;
    return { x: lerp(72, 1008, ease.inOutCubic(prog(lt, 3.1, 3.7))), y: 470 };
  }

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const lt = f.lt, t = f.t;
    const toInk = ease.inOutCubic(prog(lt, 3.9, 5.3));
    // the last walk, as a plate in the middle of the page (text above and below it)
    this.eng.render(renderer, out, this.shot.at(lt), {
      zoom: 1.02 + 0.03 * lt / 6, gain: 1.05, lift: 0.05, gamma: 1.35, angle: 0.2, mask: [0.3, 0.37, 0.64, 0.73],
      reveal: ease.outCubic(clamp(lt / 0.5)), fade: toInk,
    });
    const c = this.L.ctx;
    this.L.clear();
    const v = this.value(lt);
    c.save();
    drawCaption(c, lt, 0.2, { table: 'ТАБЛ. X', title: 'Заключение', notes: ['выборка: 1 день · 5 мест', 'погрешность: пренебрежимо мала'], alpha: 1 - prog(lt, 2.8, 3.2) });
    // the instrument, large, at the top; it gives way to the result
    const ra = 1 - prog(lt, 2.7, 3.0);
    drawReadout(c, 72, 400, v, { scale: 1.6, flash: pulse(lt, 1.5, 0.3), barW: 400, alpha: ra });
    c.font = font(F.mono(400), 22);
    c.fillStyle = rgba('bone', 0.7 * ra);
    typeOn(c, '¹ значение превышает 1.', 72, 500, lt, 1.7, 0.35);
    typeOn(c, '  методика пересматривается.', 72, 532, lt, 1.95, 0.4);
    // the result, as a numbered equation
    const ek = ease.outExpo(prog(lt, 3.0, 3.6));
    c.save();
    c.beginPath(); c.rect(0, 320, W, 140); c.clip();
    c.font = font(F.serif(600, true), 104);
    c.fillStyle = rgba('bone', 0.98);
    c.fillText('P(вернуться) = 1', 72, 435 + (1 - ek) * 130);
    c.font = font(F.serif(500, false), 60);
    c.textAlign = 'right';
    c.fillStyle = rgba('bone', 0.8);
    c.fillText('(1)', 1008, 435 + (1 - ek) * 130);
    c.restore();
    const hd = this.head(lt);
    if (hd) { c.fillStyle = rgba('bone', 0.85); c.fillRect(72, 469, hd.x - 72, 1.6); }
    c.font = font(F.mono(500), 21);
    c.letterSpacing = '5px';
    c.fillStyle = rgba('bone', 0.7);
    typeOn(c, 'Ч. Т. Д.', 72, 530, lt, 3.8, 0.25);
    c.letterSpacing = '0px';
    c.font = font(F.serif(500, true), 36);
    c.fillStyle = rgba('bone', 0.65);
    typeOn(c, 'Армения, 27 сентября. Конец трактата.', 72, 1500, lt, 4.1, 0.6);
    c.restore();
    comp.draw(renderer, this.L.upload(), out);
    const lb = this.lines;
    lb.clear();
    if (hd && lt < 5.4) {
      sparkParticles(lb, t, (tb) => this.head(tb - f.start), { rate: 60, intensity: 0.8, life: 0.35, speed: 160 });
      sparkHead(lb, hd.x, hd.y, t, 0.9, 1 - prog(lt, 5.0, 5.4));
    }
    lb.render(renderer, out);
    void LIN;
    return {
      paper: 0, bloom: 0.55, vignette: 0.4,
      frame: ease.outExpo(prog(lt, 3.9, 4.8)),
      flash: 0.3 * pulse(lt, 0, 0.12) + 0.25 * pulse(lt, 1.5, 0.12) + 0.2 * pulse(lt, 3.0, 0.1),
      fade: ease.inQuad(prog(lt, 5.1, 6.0)),
    };
  }
}
