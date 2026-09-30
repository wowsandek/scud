# Один день в Армении — иллюстрированный трактат

A 46-second vertical (1080×1920) video of a day trip from Yerevan to Garni, Geghard and Lake Sevan. It is
styled as an illustrated treatise: the footage is engraved into line plates, a contour map follows the
route, and a deadpan "P(восторг)" instrument rises from place to place.

The engine is a fork of [mexicat/pdoom-video](https://github.com/mexicat/pdoom-video) (MIT, see
`LICENSE`; upstream notes in `docs/ENGINE.upstream.md`). It is TypeScript + three.js, and every frame is a
deterministic function of time, rendered headless and piped to ffmpeg. Only the engine code comes from
upstream. The song, lyrics, data and scenes here are this project's own.

Changes from upstream:

- **Format:** vertical 1080×1920.
- **Fonts:** a Cyrillic grotesk (Roboto Flex static instances in place of Archivo).
- **Rendering:** `CHROME_PATH` support for SwiftShader.
- **Assets:** new footage, terrain and scene modules (`app/src/scenes`, `app/src/engine/footage.ts`).

## Rebuilding

The generated assets are git-ignored. Rebuild them from the source clips (`c1..c9`: 1080×1920 proxies of
the day's phone footage), the terrain tiles and the music:

```sh
python3 analysis/make_audio.py <swish-swed.wav> .   # audio/track.wav + data/audio.json (120 BPM grid)
python3 analysis/make_footage.py <proxy_dir> .      # app/public/footage/*.bin (+ index.json)
python3 analysis/make_map.py <map_dir> .            # app/public/map/terrain.bin (+ map.json, scenes/_lake.ts)
cd app && bun install
bunx vite --port 5173                               # preview player: http://localhost:5173/?t=22
CHROME_PATH=... bun scripts/render.ts video --fps 30 --samples 2 --out ../out/treatise.mp4
```

Music: "Swish Swed" (Mixkit, free licence).
