"""Blender scenes for the reel: 3D globe, satellite terrain flyover with route and
extruded labels, and a chrome 3D word.

Run: blender -b -P map3d.py -- <work_dir> <part> [frame_start frame_end] [samples] [scale%]
  part: globe | terrain | chrome
Frames are written to <work_dir>/render3d/<part>/####.png
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
WORK, PART = argv[0], argv[1]
F0 = int(argv[2]) if len(argv) > 2 else None
F1 = int(argv[3]) if len(argv) > 3 else None
SAMPLES = int(argv[4]) if len(argv) > 4 else 24
SCALE = int(argv[5]) if len(argv) > 5 else 100

MAP = os.path.join(WORK, "map")
FONTS = os.path.join(WORK, "fonts")
OUTDIR = os.path.join(WORK, "render3d", PART)
os.makedirs(OUTDIR, exist_ok=True)

# geography -------------------------------------------------------------------
LAT_C, LON_C = 40.35, 44.9
KX = 111.32 * math.cos(math.radians(LAT_C))
KY = 110.57
EXAG = 2.2
PLACES = {
    "garni": (40.112, 44.735),
    "sevanavank": (40.5645, 45.0125),
    "lake": (40.33, 45.28),
}


def xy(lat, lon):
    return (lon - LON_C) * KX, (lat - LAT_C) * KY


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_adaptive_sampling = True
    try:
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception as e:  # build without OIDN
        print("denoiser unavailable:", e)
        sc.cycles.use_denoising = False
    if os.environ.get("ENGINE") == "EEVEE":  # much faster on CPU-only machines (run under xvfb-run)
        sc.render.engine = "BLENDER_EEVEE"
        ee = sc.eevee
        ee.taa_render_samples = int(os.environ.get("EEVEE_SAMPLES", "16"))
        ee.use_soft_shadows = True
        ee.shadow_cascade_size = "4096"
        ee.use_gtao = True
        ee.gtao_distance = 2.0
        ee.use_bloom = True
        ee.bloom_intensity = 0.04
    sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
    sc.render.resolution_percentage = SCALE
    sc.render.fps = 30
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.filepath = os.path.join(OUTDIR, "####")
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.cycles.max_bounces = 4
    sc.render.use_persistent_data = True
    return sc


def principled(name, color=(1, 1, 1, 1), rough=0.5, metal=0.0, emit=None, emit_strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit is not None:
        b.inputs["Emission Color"].default_value = emit
        b.inputs["Emission Strength"].default_value = emit_strength
    return m


def grid_mesh(name, coords, uvs, rows, cols):
    """Quad grid mesh from (rows*cols, 3) coords and (rows*cols, 2) uvs."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(rows * cols)
    me.vertices.foreach_set("co", coords.astype(np.float32).ravel())
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    v0 = (r * cols + c).ravel()
    quads = np.stack([v0, v0 + cols, v0 + cols + 1, v0 + 1], 1)
    nq = len(quads)
    me.loops.add(nq * 4)
    me.loops.foreach_set("vertex_index", quads.ravel().astype(np.int32))
    me.polygons.add(nq)
    me.polygons.foreach_set("loop_start", np.arange(0, nq * 4, 4, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(nq, 4, np.int32))
    except Exception:
        pass
    me.update(calc_edges=True)
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", uvs[quads.ravel()].astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(nq, bool))
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def empty(name, loc=(0, 0, 0)):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    bpy.context.scene.collection.objects.link(e)
    return e


def camera(lens_deg=58):
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.lens_unit = "FOV"
    cd.angle = math.radians(lens_deg)
    cd.clip_start = 0.01
    cd.clip_end = 2000
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def track(obj, target, up="UP_Y"):
    c = obj.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = up


def key(obj, frame, **props):
    for k, v in props.items():
        setattr(obj, k, v)
        obj.keyframe_insert(data_path=k, frame=frame)


def ease_all(obj, interp="BEZIER", easing="EASE_IN_OUT"):
    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = interp
                kp.easing = easing


def text_obj(body, font_path, size, extrude, bevel, mat, name):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = bpy.data.fonts.load(font_path)
    cu.size = size
    cu.extrude = extrude
    cu.bevel_depth = bevel
    cu.bevel_resolution = 3
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def world_color(sc, rgb, strength=1.0):
    w = bpy.data.worlds.new("world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*rgb, 1)
    bg.inputs["Strength"].default_value = strength
    sc.world = w
    return w


def frames(sc, a, b):
    sc.frame_start = F0 if F0 is not None else a
    sc.frame_end = F1 if F1 is not None else b


# =============================================================================
def build_globe():
    sc = reset_scene()
    sc.render.film_transparent = False
    rows, cols = 181, 361
    lat = np.linspace(90, -90, rows)
    lon = np.linspace(-180, 180, cols)
    LA, LO = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
    co = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    uv = np.stack([(np.degrees(LO) + 180) / 360, (np.degrees(LA) + 90) / 180], -1).reshape(-1, 2)
    earth = grid_mesh("earth", co, uv, rows, cols)
    m = bpy.data.materials.new("earth")
    m.use_nodes = True
    nt = m.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(MAP, "bluemarble.jpg"))
    tex.interpolation = "Cubic"
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = 1.25
    hsv.inputs["Value"].default_value = 1.15
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.8
    nt.links.new(tex.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], b.inputs["Base Color"])
    earth.data.materials.append(m)
    # atmosphere rim
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.025)
    atm = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    am = bpy.data.materials.new("atm")
    am.use_nodes = True
    am.blend_method = "BLEND"
    nt = am.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.35
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.35, 0.6, 1.0, 1)
    em.inputs["Strength"].default_value = 2.5
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Facing"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    atm.data.materials.append(am)
    # plain black space; stars are added in 2D by the compositor (much faster)
    world_color(sc, (0, 0, 0), 1)
    # sun
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4.0
    so = bpy.data.objects.new("sun", sun)
    sc.collection.objects.link(so)
    # camera looks at Armenia; starts far west, swings in and dives
    la, lo = math.radians(40.35), math.radians(44.9)
    arm = Vector((math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)))
    so.rotation_mode = "QUATERNION"
    so.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(-(arm + Vector((0.3, -0.5, 0.4))).normalized())
    cam = camera(40)
    tgt = empty("tgt", arm * 1.0)
    track(cam, tgt, "UP_Y")

    def cam_pos(lat_d, lon_d, dist):
        a, o = math.radians(lat_d), math.radians(lon_d)
        return Vector((math.cos(a) * math.cos(o), math.cos(a) * math.sin(o), math.sin(a))) * dist

    key(cam, 0, location=cam_pos(25, 5, 4.2))
    key(cam, 30, location=cam_pos(34, 32, 2.4))
    key(cam, 52, location=cam_pos(39.2, 44.0, 1.25))
    key(cam, 60, location=cam_pos(40.1, 44.8, 1.08))
    key(tgt, 0, location=cam_pos(20, 20, 0.0))
    key(tgt, 30, location=cam_pos(40.35, 44.9, 0.6))
    key(tgt, 52, location=cam_pos(40.35, 44.9, 1.0))
    key(tgt, 60, location=cam_pos(40.35, 44.9, 1.0))
    ease_all(cam)
    ease_all(tgt)
    frames(sc, 0, 60)


# =============================================================================
def terrain_height_fn(h, meta):
    rows, cols = h.shape

    def f(lat, lon):
        r = (meta["lat1"] - lat) / (meta["lat1"] - meta["lat0"]) * (rows - 1)
        c = (lon - meta["lon0"]) / (meta["lon1"] - meta["lon0"]) * (cols - 1)
        r0, c0 = int(np.clip(r, 0, rows - 2)), int(np.clip(c, 0, cols - 2))
        fr, fc = r - r0, c - c0
        v = (h[r0, c0] * (1 - fr) * (1 - fc) + h[r0 + 1, c0] * fr * (1 - fc)
             + h[r0, c0 + 1] * (1 - fr) * fc + h[r0 + 1, c0 + 1] * fr * fc)
        return (v - 1000) / 1000 * EXAG
    return f


PLACES.update({
    "yerevan": (40.1776, 44.5126),
    "garni_temple": (40.1125, 44.7302),
    "geghard": (40.1403, 44.8183),
    "sevan_beach": (40.558, 45.004),
})


def _terrain_setup():
    """Satellite terrain, sky, sun, camera and a pin factory shared by the map scenes."""
    sc = reset_scene()
    meta = json.load(open(os.path.join(MAP, "meta.json")))
    h = np.load(os.path.join(MAP, "heights.npy"))
    step = 2
    hs = h[::step, ::step]
    rows, cols = hs.shape
    lat = np.linspace(meta["lat1"], meta["lat0"], rows)
    lon = np.linspace(meta["lon0"], meta["lon1"], cols)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    X, Y = xy(LA, LO)
    Z = (hs - 1000) / 1000 * EXAG
    co = np.stack([X, Y, Z], -1).reshape(-1, 3)
    uv = np.stack([(LO - meta["lon0"]) / (meta["lon1"] - meta["lon0"]),
                   (LA - meta["lat0"]) / (meta["lat1"] - meta["lat0"])], -1).reshape(-1, 2)
    ter = grid_mesh("terrain", co, uv, rows, cols)
    m = bpy.data.materials.new("sat")
    m.use_nodes = True
    nt = m.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(MAP, "sat.jpg"))
    tex.interpolation = "Cubic"
    tex.extension = "EXTEND"
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = 1.2
    hsv.inputs["Value"].default_value = 1.35
    bc = nt.nodes.new("ShaderNodeBrightContrast")
    bc.inputs["Contrast"].default_value = 0.15
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.95
    nt.links.new(tex.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bc.inputs["Color"])
    nt.links.new(bc.outputs["Color"], b.inputs["Base Color"])
    ter.data.materials.append(m)

    haze = (0.62, 0.74, 0.88)
    world_color(sc, haze, 0.9)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.2
    sun.angle = math.radians(1.5)
    sun.color = (1.0, 0.93, 0.82)
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(58), 0, math.radians(-120))  # low sun from the south-west
    sun.shadow_cascade_max_distance = 400
    sun.shadow_cascade_count = 4
    sc.collection.objects.link(so)

    hf = terrain_height_fn(h, meta)
    pin_mat = principled("pin", (1, 0.45, 0.08, 1), 0.3, emit=(1.0, 0.42, 0.06, 1), emit_strength=2.0)
    beam_mat = principled("beam", (1, 1, 1, 1), 0.3, emit=(1, 0.85, 0.7, 1), emit_strength=1.2)
    label_mat = principled("label", (0.96, 0.96, 0.96, 1), 0.35, emit=(1, 1, 1, 1), emit_strength=0.6)
    font = os.path.join(FONTS, "Inter-ExtraBold.ttf")
    cam = camera(58)

    def pin(name, x, y, z, f_pop, text, size=2.6, f_hide=None, lift=3.6):
        root = empty(name, (x, y, z))
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 0.1))
        ball = bpy.context.active_object
        bpy.ops.object.shade_smooth()
        ball.data.materials.append(pin_mat)
        ball.parent = root
        bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=2.4, location=(0, 0, 1.2))
        beam = bpy.context.active_object
        beam.data.materials.append(beam_mat)
        beam.parent = root
        lab_root = empty(name + "_lab", (x, y, z + lift))
        c = lab_root.constraints.new("LOCKED_TRACK")
        c.target = cam
        c.track_axis = "TRACK_Y"
        c.lock_axis = "LOCK_Z"
        t = text_obj(text, font, size, 0.35, 0.05, label_mat, name + "_text")
        t.parent = lab_root
        t.rotation_euler = (math.radians(90), 0, math.radians(180))
        for ob, f in ((root, f_pop), (lab_root, f_pop + 3)):
            key(ob, 0, scale=(0, 0, 0))
            key(ob, f, scale=(0.001, 0.001, 0.001))
            key(ob, f + 6, scale=(1.18, 1.18, 1.18))
            key(ob, f + 10, scale=(0.94, 0.94, 0.94))
            key(ob, f + 14, scale=(1, 1, 1))
        if f_hide is not None:  # the label ducks away as the camera dives in
            key(lab_root, f_hide - 6, scale=(1, 1, 1))
            key(lab_root, f_hide, scale=(0.001, 0.001, 0.001))
        return root

    # mist toward the haze colour hides the edges of the tile area
    sc.view_layers[0].use_pass_mist = True
    sc.world.mist_settings.falloff = "QUADRATIC"
    sc.use_nodes = True
    nt = sc.node_tree
    rl = nt.nodes["Render Layers"]
    mix = nt.nodes.new("CompositorNodeMixRGB")
    mix.inputs[2].default_value = (*haze, 1)
    nt.links.new(rl.outputs["Mist"], mix.inputs["Fac"])
    nt.links.new(rl.outputs["Image"], mix.inputs[1])
    nt.links.new(mix.outputs[0], nt.nodes["Composite"].inputs["Image"])
    sc.render.image_settings.color_mode = "RGB"
    return sc, hf, cam, pin, font


def mist_keys(sc, keys):
    ms = sc.world.mist_settings
    for f, start, depth in keys:
        ms.start, ms.depth = start, depth
        ms.keyframe_insert("start", frame=f)
        ms.keyframe_insert("depth", frame=f)


def route_arc(hf, a, b, lift, n=90):
    pts = []
    za, zb = hf(*a), hf(*b)
    for i in range(n):
        u = i / (n - 1)
        lat = a[0] + (b[0] - a[0]) * u
        lon = a[1] + (b[1] - a[1]) * u
        x, y = xy(lat, lon)
        z = max(za + (zb - za) * u + 0.25 + lift * math.sin(math.pi * u), hf(lat, lon) + 0.35)
        pts.append((x, y, z))
    return pts


def build_terrain():
    sc, hf, cam, pin, font = _terrain_setup()
    gx, gy = xy(*PLACES["garni"])
    sx, sy = xy(*PLACES["sevanavank"])
    gz, sz = hf(*PLACES["garni"]), hf(*PLACES["sevanavank"])

    # route: glowing arc Garni -> Sevanavank
    pts = route_arc(hf, PLACES["garni"], PLACES["sevanavank"], 2.2, 120)
    cu = bpy.data.curves.new("route", "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1)
    cu.bevel_depth = 0.11
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    route = bpy.data.objects.new("route", cu)
    sc.collection.objects.link(route)
    route.data.materials.append(principled("route", (1, 0.35, 0.04, 1), 0.3, emit=(1.0, 0.33, 0.04, 1),
                                           emit_strength=1.6))
    cu.bevel_factor_end = 0.0
    cu.keyframe_insert("bevel_factor_end", frame=80)
    cu.bevel_factor_end = 1.0
    cu.keyframe_insert("bevel_factor_end", frame=128)
    for kp in cu.animation_data.action.fcurves[0].keyframe_points:
        kp.interpolation = "SINE"
        kp.easing = "EASE_IN_OUT"

    pin("garni", gx, gy, gz, 78, "ГАРНИ")
    pin("sevanavank", sx, sy, sz, 126, "СЕВАНАВАНК")
    lake_label(sc, hf, font, 98)

    # camera path (map timeline frames: terrain visible from ~45)
    cx, cy = (gx + sx) / 2, (gy + sy) / 2
    tgt = empty("tgt", (cx, cy, 0))
    track(cam, tgt)
    key(cam, 45, location=(cx, cy - 12, 120))
    key(cam, 72, location=(cx + 2, cy - 30, 80))
    key(cam, 112, location=(cx - 14, cy - 50, 36))
    key(cam, 150, location=(gx - 7, gy - 24, gz + 15))
    key(cam, 180, location=(gx - 1.0, gy - 7.5, gz + 4.2))
    key(tgt, 45, location=(cx, cy, 0))
    key(tgt, 72, location=(cx, cy + 4, 0))
    key(tgt, 112, location=(cx + 2, cy + 4, 2))
    key(tgt, 150, location=(gx, gy + 2, gz))
    key(tgt, 180, location=(gx, gy + 1.0, gz + 1.2))
    ease_all(cam)
    ease_all(tgt)
    mist_keys(sc, ((45, 400, 400), (70, 400, 400), (105, 45, 110)))
    frames(sc, 45, 180)


def lake_label(sc, hf, font, f_pop):
    lx, ly = xy(*PLACES["lake"])
    lz = hf(*PLACES["lake"]) + 0.05
    lake_mat = principled("lake", (1, 1, 1, 1), 0.3, emit=(1, 1, 1, 1), emit_strength=1.2)
    lake = text_obj("оз. Севан", font, 5.0, 0.12, 0.03, lake_mat, "lake")
    lake.location = (lx, ly, lz)
    lake.rotation_euler = (0, 0, math.radians(-14))
    key(lake, 0, scale=(0, 0, 0))
    key(lake, f_pop, scale=(0.001, 0.001, 0.001))
    key(lake, f_pop + 10, scale=(1.08, 1.08, 1.08))
    key(lake, f_pop + 16, scale=(1, 1, 1))


# frame ranges of the four flights in the "route" scene
ROUTE_SEGMENTS = {"A": (45, 90), "B": (1000, 1059), "C": (2000, 2044), "D": (3000, 3089)}


def build_route():
    """Whole-day route: Yerevan -> Garni -> Geghard -> Sevan, flown in four separate shots."""
    sc, hf, cam, pin, font = _terrain_setup()
    P = {k: (*xy(*PLACES[k]), hf(*PLACES[k])) for k in ("yerevan", "garni_temple", "geghard", "sevan_beach")}
    Yv, G, K, S = (Vector(P[k]) for k in ("yerevan", "garni_temple", "geghard", "sevan_beach"))

    legs = [route_arc(hf, PLACES["yerevan"], PLACES["garni_temple"], 1.8),
            route_arc(hf, PLACES["garni_temple"], PLACES["geghard"], 0.9, 50),
            route_arc(hf, PLACES["geghard"], PLACES["sevan_beach"], 4.0, 140)]
    pts = legs[0] + legs[1][1:] + legs[2][1:]
    seglen = [0.0]
    for p0, p1 in zip(pts, pts[1:]):
        seglen.append(seglen[-1] + (Vector(p1) - Vector(p0)).length)
    total = seglen[-1]
    r1 = seglen[len(legs[0]) - 1] / total
    r2 = seglen[len(legs[0]) + len(legs[1]) - 2] / total
    cu = bpy.data.curves.new("route", "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1)
    cu.bevel_depth = 0.09
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    cu.bevel_factor_mapping_end = "SPLINE"
    route = bpy.data.objects.new("route", cu)
    sc.collection.objects.link(route)
    route.data.materials.append(principled("route", (1, 0.35, 0.04, 1), 0.3, emit=(1.0, 0.33, 0.04, 1),
                                           emit_strength=1.6))
    for f, v in ((0, 0.0), (1004, 0.0), (1036, r1), (2003, r1), (2022, r2), (3008, r2), (3062, 1.0)):
        cu.bevel_factor_end = v
        cu.keyframe_insert("bevel_factor_end", frame=f)
    for kp in cu.animation_data.action.fcurves[0].keyframe_points:
        kp.interpolation = "SINE"
        kp.easing = "EASE_IN_OUT"

    pin("yerevan", *Yv, 60, "ЕРЕВАН", f_hide=86)
    pin("garni", *G, 1036, "ГАРНИ", size=1.9, f_hide=1057, lift=2.5)
    pin("geghard", *K, 2022, "ГЕГАРД", size=1.9, f_hide=2043, lift=2.5)
    pin("sevan", *S, 3062, "СЕВАН", f_hide=3087)
    lake_label(sc, hf, font, 3032)

    tgt = empty("tgt", (0, 0, 0))
    track(cam, tgt)

    def shot(keys):
        """keys: [(frame, cam_pos, target_pos)] for one flight; eased inside, cut at the ends."""
        for f, c, t in keys:
            key(cam, f, location=tuple(c))
            key(tgt, f, location=tuple(t))

    def mid(a, b, k=0.5):
        return a + (b - a) * k

    V = Vector
    shot([(45, V((0, -12, 120)), V((0, 0, 0))),
          (68, Yv + V((-6, -24, 18)), Yv),
          (90, Yv + V((-0.8, -4.5, 1.7)), Yv + V((0, 0, 0.2)))])
    shot([(1000, Yv + V((-7, -15, 13)), mid(Yv, G, 0.3)),
          (1032, mid(Yv, G, 0.55) + V((-3, -15, 10)), G),
          (1059, G + V((-0.8, -4.5, 1.6)), G + V((0, 0, 0.2)))])
    shot([(2000, G + V((-4, -10, 7)), mid(G, K, 0.4)),
          (2024, mid(G, K, 0.6) + V((-2, -9, 5.5)), K),
          (2044, K + V((-0.8, -4.5, 1.6)), K + V((0, 0, 0.2)))])
    shot([(3000, K + V((-4, -11, 8)), K + V((3, 8, 0))),
          (3030, mid(K, S, 0.25) + V((-10, -22, 30)), mid(K, S, 0.6)),
          (3066, S + V((-6, -24, 16)), S),
          (3089, S + V((-0.8, -4.8, 1.7)), S + V((0, 0, 0.2)))])
    for ob in (cam, tgt):
        for fc in ob.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
                kp.easing = "EASE_IN_OUT"
                kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
                if int(kp.co[0]) in (90, 1059, 2044):
                    kp.interpolation = "CONSTANT"   # hard cut to the next flight
    mist_keys(sc, ((45, 400, 400), (66, 400, 400), (88, 45, 110)))
    seg = os.environ.get("SEG", "A")
    a, b = ROUTE_SEGMENTS[seg]
    frames(sc, a, b)


# =============================================================================
def build_chrome():
    sc = reset_scene()
    sc.render.film_transparent = True
    w = world_color(sc, (1, 1, 1), 1.0)
    nt = w.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(os.path.join(MAP, "sky.hdr"))
    nt.links.new(env.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 1.4
    mat = principled("chrome", (0.95, 0.96, 1.0, 1), 0.07, 1.0)
    t = text_obj("ГИДРОЦИКЛ", os.path.join(FONTS, "Unbounded-Black.ttf"), 1.0, 0.22, 0.035, mat, "word")
    t.data.bevel_resolution = 4
    key_light = bpy.data.lights.new("key", "AREA")
    key_light.energy = 900
    key_light.size = 6
    kl = bpy.data.objects.new("key", key_light)
    kl.location = (3, -4, 5)
    sc.collection.objects.link(kl)
    track(kl, t)
    rim = bpy.data.lights.new("rim", "AREA")
    rim.energy = 700
    rim.color = (1.0, 0.6, 0.25)
    rim.size = 4
    rl = bpy.data.objects.new("rim", rim)
    rl.location = (-4, 3, 2)
    sc.collection.objects.link(rl)
    track(rl, t)
    cam = camera(40)
    cam.location = (0, -16.5, 0)
    tgt = empty("tgt", (0, 0, 0))
    track(cam, tgt, "UP_Y")
    t.rotation_mode = "XYZ"
    # fly in from depth while spinning, settle, float, then rip past the camera
    key(t, 0, location=(0.8, 40, 3.0), rotation_euler=(math.radians(90 + 25), 0, math.radians(-120)))
    key(t, 16, location=(0, -0.6, 0), rotation_euler=(math.radians(90 - 6), 0, math.radians(6)))
    key(t, 22, location=(0, 0, 0), rotation_euler=(math.radians(90), 0, 0))
    key(t, 56, location=(0, -0.4, 0.1), rotation_euler=(math.radians(90 + 4), 0, math.radians(-7)))
    key(t, 72, location=(-0.6, -13.5, -0.4), rotation_euler=(math.radians(90 - 30), 0, math.radians(70)))
    ease_all(t)
    for fc in t.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            if kp.co[0] == 72:
                kp.easing = "EASE_IN"
            if kp.co[0] == 16:
                kp.easing = "EASE_OUT"
    frames(sc, 0, 72)


if __name__ == "__main__":
    {"globe": build_globe, "terrain": build_terrain, "route": build_route, "chrome": build_chrome}[PART]()
    bpy.ops.render.render(animation=True)
