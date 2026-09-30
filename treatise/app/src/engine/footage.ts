// Pre-cut footage (app/public/footage, made by analysis/make_footage.py): per shot, the frames the
// edit needs at 30 fps as a 2-channel raster (R = luminance with local contrast, G = the "signal" mask:
// orange/red surfaces and sun glints). One RG8 DataTexture per shot is refilled when the frame changes.
import * as THREE from 'three';

export interface ShotMeta { frames: number; w: number; h: number; fps: number; clip: string; in: number; sig?: [number, number, number][] }

let INDEX: Record<string, ShotMeta> | null = null;
async function index() {
  if (!INDEX) INDEX = await (await fetch('footage/index.json')).json();
  return INDEX!;
}

export class Shot {
  tex: THREE.DataTexture;
  private cur = -1;
  constructor(public id: string, public meta: ShotMeta, private data: Uint8Array) {
    const { w, h } = meta;
    this.tex = new THREE.DataTexture(new Uint8Array(w * h * 2), w, h, THREE.RGFormat, THREE.UnsignedByteType);
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.generateMipmaps = false;
    this.tex.flipY = false; // rows are top-first: the shader flips v
    this.set(0);
  }
  get duration() { return this.meta.frames / this.meta.fps; }
  /** Frame index shown at local time lt (clamped; constant over a 30 fps frame). */
  frameAt(lt: number) { return Math.max(0, Math.min(this.meta.frames - 1, Math.floor(lt * this.meta.fps + 1e-6))); }
  set(i: number) {
    i = Math.max(0, Math.min(this.meta.frames - 1, i));
    if (i === this.cur) return this.tex;
    const n = this.meta.w * this.meta.h * 2;
    (this.tex.image.data as Uint8Array).set(this.data.subarray(i * n, (i + 1) * n));
    this.tex.needsUpdate = true;
    this.cur = i;
    return this.tex;
  }
  at(lt: number) { return this.set(this.frameAt(lt)); }
  /** Centroid (0..1, y down) and area of the signal mask at local time lt, if precomputed. */
  signalAt(lt: number): [number, number, number] | null {
    return this.meta.sig ? this.meta.sig[this.frameAt(lt)] ?? null : null;
  }
}

export async function loadShot(id: string): Promise<Shot> {
  const meta = (await index())[id];
  if (!meta) throw new Error(`footage shot not found: ${id}`);
  const buf = new Uint8Array(await (await fetch(`footage/${id}.bin`)).arrayBuffer());
  return new Shot(id, meta, buf);
}
