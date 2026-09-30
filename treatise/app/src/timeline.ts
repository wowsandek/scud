// The edit: one day in Armenia as an illustrated treatise. Boundaries sit on the beat grid of the
// track (120 BPM: a bar every 2 s); the drop (22.0 s, beat 44) is the arrival at Lake Sevan.
import type { TimelineEntry } from './engine/engine';
import type { SceneClass } from './engine/scene';
import type { Lyrics } from './engine/lyrics';
import type { AudioData } from './engine/audio';

// Scene modules are discovered lazily so a missing/broken scene never breaks the build.
const modules = import.meta.glob<{ default: SceneClass }>('./scenes/*.ts');
const scene = (name: string) => () => {
  const m = modules[`./scenes/${name}.ts`];
  return m ? m() : Promise.reject(new Error(`scene module not found: scenes/${name}.ts`));
};

export function makeTimeline(_ly: Lyrics, au: AudioData): TimelineEntry[] {
  const bar = (n: number) => au.timeOfBeat(n * 4);
  const beat = (n: number) => au.timeOfBeat(n);
  const E = (id: string, file: string, start: number, end: number, extra: Partial<TimelineEntry> = {}): TimelineEntry =>
    ({ id, load: scene(file), start, end, ...extra });
  const plate = (id: string, start: number, end: number) => E(id, 'plate', start, end, { params: { id } });

  return [
    E('open', 'open', 0, bar(2)),                                   // 0–4    title page
    E('route1', 'route', bar(2), bar(3), { params: { leg: 1 } }),   // 4–6    Ереван → Гарни
    plate('garni', bar(3), bar(5)),                                 // 6–10   Табл. II
    plate('basalt', bar(5), beat(26)),                              // 10–13  Табл. III
    E('route2', 'route', beat(26), bar(7), { params: { leg: 2 } }), // 13–14  Гарни → Гегард
    plate('geghard', bar(7), bar(9)),                               // 14–18  Табл. IV
    E('route3', 'route', bar(9), bar(11), { params: { leg: 3 } }),  // 18–22  riser: → Севан
    E('hook', 'hook', bar(11), bar(12)),                            // 22–24  the drop
    plate('sevan', bar(12), bar(14)),                               // 24–28  Табл. VI
    plate('jetski', bar(14), bar(16)),                              // 28–32  Табл. VII
    plate('vank', bar(16), bar(18)),                                // 32–36  Табл. VIII
    plate('peninsula', bar(18), bar(20)),                           // 36–40  Табл. IX
    E('outro', 'outro', bar(20), au.duration),                      // 40–46  Заключение
  ];
}
