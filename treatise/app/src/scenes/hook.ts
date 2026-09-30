// The drop (22.0): the lake reached, the instrument takes the whole frame. A signal-orange field,
// the conditional probability as a label, huge digits rolling from the old value to the new one like
// an odometer, the bar jumping, a deadpan footnote. Hard cut out on the bar line (24.0).
import * as THREE from 'three';
import { Scene, type Frame, type PostOverrides } from '../engine/scene';
import { Layer2D, W, clearRT } from '../engine/gl';
import { LIN, rgba } from '../engine/palette';
import { F, font } from '../engine/type';
import { clamp, ease, lerp, prog, pulse } from '../engine/util';
import { typeOn } from './_plate';

export default class Hook extends Scene {
  L = new Layer2D();

  override render(f: Frame, out: THREE.WebGLRenderTarget): PostOverrides {
    const { renderer, comp } = this.ctx;
    const lt = f.lt;
    const from = 0.64, to = 0.81;
    clearRT(renderer, out, [LIN.signal[0] * 0.9, LIN.signal[1] * 0.9, LIN.signal[2] * 0.9]);
    const c = this.L.ctx;
    this.L.clear();
    c.save();
    c.textBaseline = 'alphabetic';
    // label
    c.font = font(F.mono(500), 30);
    c.letterSpacing = '8px';
    c.fillStyle = rgba('ink', 0.9);
    typeOn(c, 'P(ВОСТОРГ | ОЗЕРО)', 80, 640, lt, 0.0, 0.3);
    c.letterSpacing = '0px';
    // the digits, rolling: each place is a reel; hundredths spin more than tenths
    const k = ease.outExpo(prog(lt, 0.05, 0.9));
    const v = lerp(from, to, k);
    const size = 400;
    c.font = font(F.archivo(1250, 900), size);
    const dw = c.measureText('0').width;
    const x0 = 64, base = 1060;
    c.fillStyle = rgba('ink', 0.97);
    c.fillText('0.', x0, base);
    const xd = x0 + c.measureText('0.').width;
    const reels = [v * 10, v * 100];
    reels.forEach((r, i) => {
      const x = xd + i * dw;
      c.save();
      c.beginPath(); c.rect(x - 4, base - size * 0.8, dw + 8, size * 0.86); c.clip();
      const d = Math.floor(r), fr = r - d;
      for (let j = -1; j <= 1; j++) {
        const digit = ((d + j) % 10 + 10) % 10;
        c.fillText(String(digit), x, base - (j - fr) * size * 0.95 * -1);
      }
      c.restore();
    });
    // rules above and below, like an instrument's bezel
    c.fillStyle = rgba('ink', 0.85);
    c.fillRect(80, 700, (W - 160) * ease.outExpo(prog(lt, 0, 0.5)), 3);
    // the bar: 0..1 with ticks, the fill jumps
    const by = 1180, bw = W - 160;
    c.fillRect(80, by, bw, 2);
    for (let i = 0; i <= 10; i++) c.fillRect(80 + (bw * i) / 10, by - (i % 5 ? 12 : 22), 2, i % 5 ? 12 : 22);
    c.fillStyle = rgba('ink', 0.95);
    c.fillRect(80, by - 6, bw * v, 14);
    c.font = font(F.mono(400), 22);
    c.fillStyle = rgba('ink', 0.75);
    c.fillText('0', 80, by + 44);
    c.textAlign = 'right';
    c.fillText('1', 80 + bw, by + 44);
    c.textAlign = 'left';
    // footnotes
    c.font = font(F.mono(400), 25);
    c.fillStyle = rgba('ink', 0.85);
    typeOn(c, `было ${from.toFixed(2)} · стало ${to.toFixed(2)}`, 80, 1320, lt, 0.5, 0.35);
    typeOn(c, 'апостериорно · обновлено озером (1 шт.)', 80, 1362, lt, 0.8, 0.45);
    c.font = font(F.serif(600, true), 64);
    c.fillStyle = rgba('ink', 0.95);
    const tk = ease.outExpo(prog(lt, 1.0, 1.5));
    c.save();
    c.beginPath(); c.rect(60, 1450, W, 90); c.clip();
    c.fillText('Прибыли на Севан.', 80, 1520 + (1 - tk) * 80);
    c.restore();
    c.restore();
    comp.draw(renderer, this.L.upload(), out);
    const hit = pulse(lt, 0, 0.12), hit2 = pulse(lt, 1.0, 0.1);
    return {
      paper: 1, bloom: 0.35, vignette: 0.25, grain: 0.07,
      flash: 1.1 * hit + 0.3 * hit2, zoom: 1 + 0.05 * hit + 0.015 * hit2,
      shake: [Math.sin(lt * 91) * 10 * hit, Math.cos(lt * 77) * 8 * hit],
    };
  }
}
