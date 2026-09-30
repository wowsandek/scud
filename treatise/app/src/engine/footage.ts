// Pre-cut footage (app/public/footage, made by analysis/make_footage.py), per shot the frames the edit
// needs at 30 fps, twice:
//  - Shot: a 2-channel raster (R = luminance with local contrast, G = the "signal" mask: orange/red
//    surfaces and glints) for the engraving; one RG8 DataTexture per shot, refilled when the frame changes.
//  - PhotoShot: the natural-colour frames, full size, as JPEGs streamed per frame (Scene.prepare loads
//    them before the export renders a frame; the preview uses whatever has arrived).
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

/** How many decoded frames a PhotoShot keeps (8 MB each at 1080x1920). */
const KEEP = 3;

export class PhotoShot {
  tex = new THREE.Texture();
  private cache = new Map<number, ImageBitmap>();
  private pending = new Map<number, Promise<void>>();
  private cur = -1;
  constructor(public id: string, public meta: ShotMeta) {
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.generateMipmaps = false;
    this.tex.flipY = false; // rows are top-first (ImageBitmaps are not flipped on upload): the shader flips v
  }
  get duration() { return this.meta.frames / this.meta.fps; }
  frameAt(lt: number) { return Math.max(0, Math.min(this.meta.frames - 1, Math.floor(lt * this.meta.fps + 1e-6))); }
  signalAt(lt: number): [number, number, number] | null {
    return this.meta.sig ? this.meta.sig[this.frameAt(lt)] ?? null : null;
  }
  private ensure(i: number): Promise<void> {
    if (this.cache.has(i)) return Promise.resolve();
    let p = this.pending.get(i);
    if (!p) {
      p = fetch(`footage/rgb/${this.id}/${String(i).padStart(4, '0')}.jpg`)
        .then((r) => r.blob())
        .then((b) => createImageBitmap(b))
        .then((bm) => { this.cache.set(i, bm); this.pending.delete(i); });
      this.pending.set(i, p);
    }
    return p;
  }
  /** Drop decoded frames beyond KEEP, farthest from `around` first (never the one on the texture). */
  private evict(around: number) {
    if (this.cache.size <= KEEP) return;
    const ks = [...this.cache.keys()].filter((k) => k !== this.cur).sort((a, b) => Math.abs(b - around) - Math.abs(a - around));
    for (const k of ks.slice(0, this.cache.size - KEEP)) { this.cache.get(k)!.close(); this.cache.delete(k); }
  }
  async prepare(lts: number[]) {
    const need = [...new Set(lts.map((lt) => this.frameAt(lt)))];
    await Promise.all(need.map((i) => this.ensure(i)));
    this.evict(need[0]!);
  }
  /** The texture showing the frame at lt (or the nearest one loaded so far, while it streams in). */
  at(lt: number) {
    let i = this.frameAt(lt);
    if (!this.cache.has(i)) {
      void this.ensure(i);
      for (let k = 1; k <= 3; k++) void this.ensure(Math.min(this.meta.frames - 1, i + k));
      let best = -1;
      for (const k of this.cache.keys()) if (best < 0 || Math.abs(k - i) < Math.abs(best - i)) best = k;
      i = best;
    }
    if (i >= 0 && i !== this.cur) {
      this.tex.image = this.cache.get(i)!;
      this.tex.needsUpdate = true;
      this.cur = i;
    }
    return this.tex;
  }
}

export async function loadPhoto(id: string): Promise<PhotoShot> {
  const meta = (await index())[id];
  if (!meta) throw new Error(`footage shot not found: ${id}`);
  const ph = new PhotoShot(id, meta);
  await ph.prepare([0]);
  ph.at(0);
  return ph;
}
