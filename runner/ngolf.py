#!/usr/bin/env python3
"""ngolf — cost-aware ONNX builder for NeuroGolf workers. IMPORT THIS, don't write protobuf.

Every primitive returns a T(name, shape, dtype) and TRACKS ITS EXACT GRADER COST as you
build. The final renderer's output is free (it becomes graph output `output`). save()
runs checker + strict shape-inference + an ORT zeros-probe and prints the budget table —
if save() passes, fast_verify's price will match what you saw.

Worked example (terminal-Equal renderer pattern):
    import sys; sys.path.insert(0, "../../runner")   # build.py runs from packs/taskNNN/
    from ngolf import G
    g = G(task=364)
    m   = g.color_plane(3, cast="u8")          # one-hot ch3 == mask of color 3 (2 ops)
    d   = g.maxpool(m, 3)                       # dilate
    d   = g.mul(d, m)                           # geodesic constraint
    st  = g.pad_to(d, 30, 30)                   # back to 30x30 canvas, uint8
    g.out_equal_arange(st, K=10)                # FREE terminal render -> [1,10,30,30] bool
    print(g.budget())                           # predicted params/memory/cost/points
    g.save("task364.onnx")

Key idioms as one-liners: recolor = g.channel_permute(perm) · bbox mask = g.bbox_mask(m)
· color-c mask = g.color_plane(c) (the input is ONE-HOT: channel c IS the mask).
"""
import math
import numpy as np
import onnx
from onnx import helper as H, TensorProto as TP

DT = {"f32": (TP.FLOAT, 4, np.float32), "f16": (TP.FLOAT16, 2, np.float16),
      "u8": (TP.UINT8, 1, np.uint8), "i8": (TP.INT8, 1, np.int8),
      "b": (TP.BOOL, 1, np.bool_), "i32": (TP.INT32, 4, np.int32),
      "u32": (TP.UINT32, 4, np.uint32), "i64": (TP.INT64, 8, np.int64)}
NP2DT = {v[2]: k for k, v in DT.items()}
RELOWER_DOC = "ngolf.relower_onehot_plane"
PAINT_BLUE_DOC = "ngolf.paint_blue_from_anchor"
PAINT_TASK119_DOC = "ngolf.paint_task119_green_cells_nchw"
TASK224_STATUS_DOC = "ngolf.task224_status_tail"
TERMINAL_HODEL_RAY_DOC = "ngolf.terminal_hodel_ray"
INDEXED_U8_SCRATCH_DOC = "ngolf.indexed_u8_scratch"
INPUT_U8_DOC = "ngolf.input_u8"
COMPACT_BBOX_U8_DOC = "ngolf.compact_bbox_u8"
GATHER_ND_U8_DOC = "ngolf.gather_nd_u8"
SCATTER_ND_U8_DOC = "ngolf.scatter_nd_u8"


def _relower_bounds(channel, crop):
    if crop is None:
        bounds = ((0, 1), (channel, channel + 1), (0, 30), (0, 30))
    elif len(crop) == 4 and all(len(p) == 2 for p in crop):
        bounds = tuple((int(a), int(b)) for a, b in crop)
    elif len(crop) == 8:
        vals = [int(v) for v in crop]
        bounds = tuple((vals[i], vals[i + 1]) for i in range(0, 8, 2))
    else:
        raise ValueError("crop must be four (lo, hi) pairs or eight flat bounds")

    (n0, n1), (c0, c1), (y0, y1), (x0, x1) = bounds
    if (n0, n1) != (0, 1):
        raise ValueError("relower_onehot_plane only supports batch crop 0:1")
    if (c0, c1) != (channel, channel + 1):
        raise ValueError("relower_onehot_plane crop must select exactly the requested channel")
    if not (0 <= channel < 10 and 0 <= y0 < y1 <= 30 and 0 <= x0 < x1 <= 30):
        raise ValueError("relower_onehot_plane crop is outside the fixed input shape")
    shape = [1, 1, y1 - y0, x1 - x0]
    if shape == [1, 10, 30, 30] or shape[-2] * shape[-1] > 900:
        raise ValueError("relower_onehot_plane refuses full-frame relowers")
    return bounds, shape


def relower_onehot_plane(input_name="input", output_name="relower_b", *, channel=0,
                         crop=None, dtype="b", starts_name="relower_starts",
                         ends_name="relower_ends"):
    """Crop+dtype relower for one one-hot channel (Slice fixed crop, Cast to bool/u8).

    ⚠️ COST (fixed 2026-07-10): the transient float Slice output IS charged by the real
    Kaggle grader — crop_elems×4 B on top of the crop_elems×1 B u8/bool result. The old
    claim that a "tagged" relower is free came from a locally-hacked scorer and cost real
    LB points (#p807). Relowering still pays off when downstream work reuses the small u8
    plane many times; it is NOT free. save() auto-charges the transient.
    """
    if dtype not in ("b", "u8"):
        raise ValueError("relower_onehot_plane dtype must be 'b' or 'u8'")
    bounds, shape = _relower_bounds(channel, crop)
    starts = np.array([lo for lo, _ in bounds], dtype=np.int64)
    ends = np.array([hi for _, hi in bounds], dtype=np.int64)
    tmp = f"__ngolf_relower_onehot_plane__{output_name}__f32"
    sl = H.make_node("Slice", [input_name, starts_name, ends_name], [tmp])
    sl.doc_string = f"{RELOWER_DOC};channel={channel};shape={','.join(map(str, shape))};dtype={dtype}"
    ca = H.make_node("Cast", [tmp], [output_name], to=DT[dtype][0])
    ca.doc_string = RELOWER_DOC
    return [sl, ca], [(starts_name, starts), (ends_name, ends)], shape


def _paint_bounds(crop):
    if crop is None:
        bounds = ((0, 20), (0, 20))
    elif len(crop) == 2 and all(len(p) == 2 for p in crop):
        bounds = tuple((int(a), int(b)) for a, b in crop)
    elif len(crop) == 4:
        vals = [int(v) for v in crop]
        bounds = ((vals[0], vals[1]), (vals[2], vals[3]))
    else:
        raise ValueError("crop must be two (lo, hi) pairs or four flat y/x bounds")
    (y0, y1), (x0, x1) = bounds
    if (y0, y1, x0, x1) != (0, 20, 0, 20):
        raise ValueError("paint_blue_from_anchor only supports the certified 20x20 crop")
    return bounds


def paint_blue_from_anchor(input_name="input", anchor_name="anchor_u8", output_name="output",
                           *, crop=None, color=1, stamp_name="stamp_u8",
                           mask20_name="mask20_b", mask30_name="mask30_b",
                           pads_name="paint_pads", axes_name="paint_axes",
                           blue_name="blue_onehot"):
    """Terminal painter for task162's anchor mask.

    Emits executable ONNX for MaxPool(anchor)->Cast(bool)->Pad(30x30)->Where(blue,input).
    Every intermediate output is charged by the fixed scorer; doc_string tags are cosmetic.
    """
    _paint_bounds(crop)
    if color != 1:
        raise ValueError("paint_blue_from_anchor currently only supports blue color=1")
    blue = np.zeros((1, 10, 1, 1), dtype=np.float32)
    blue[0, color, 0, 0] = 1.0
    tag = f"{PAINT_BLUE_DOC};color={color};crop=0,20,0,20"

    pool = H.make_node(
        "MaxPool",
        [anchor_name],
        [stamp_name],
        kernel_shape=[3, 3],
        pads=[2, 2, 2, 2],
        strides=[1, 1],
    )
    pool.doc_string = tag
    cast = H.make_node("Cast", [stamp_name], [mask20_name], to=TP.BOOL)
    cast.doc_string = tag
    # Axes-form Pad is equivalent to full pads [0,0,0,0,0,0,10,10] and preserves
    # the att09 44-param budget.
    pad = H.make_node("Pad", [mask20_name, pads_name, "", axes_name], [mask30_name],
                      mode="constant")
    pad.doc_string = tag
    where = H.make_node("Where", [mask30_name, blue_name, input_name], [output_name])
    where.doc_string = tag

    inits = [
        (pads_name, np.array([0, 0, 10, 10], dtype=np.int64)),
        (axes_name, np.array([2, 3], dtype=np.int64)),
        (blue_name, blue),
    ]
    return [pool, cast, pad, where], inits, [1, 10, 30, 30]


def paint_task119_green_cells_nchw(input_name="input", coords_name="coords_u8",
                                   valid_name="valid_b", output_name="output",
                                   *, color=3, batch_name="paint119_batch_u8",
                                   black_name="paint119_black_u8",
                                   green_name="paint119_green_u8",
                                   black_idx_name="paint119_black_idx_u8",
                                   green_idx_name="paint119_green_idx_u8",
                                   idx_u8_name="paint119_idx_u8",
                                   idx_i64_name="paint119_idx_i64",
                                   valid_f_name="paint119_valid_f",
                                   neg_valid_name="paint119_neg_valid_f",
                                   updates_name="paint119_updates_f"):
    """Terminal painter for task119's hidden green V cells.

    Executable lowering: build rank-4 ScatterND indices from compact [row,col]
    coords and a [10] valid mask, then add -1 to black and +1 to green.  The
    Every intermediate output is charged by the fixed scorer, including the index plumbing.
    """
    if color != 3:
        raise ValueError("paint_task119_green_cells_nchw currently supports ARC green color=3")
    tag = f"{PAINT_TASK119_DOC};color={color}"
    nodes = [
        H.make_node("Cast", [valid_name], [valid_f_name], to=TP.FLOAT),
        H.make_node("Neg", [valid_f_name], [neg_valid_name]),
        H.make_node("Concat", [neg_valid_name, valid_f_name], [updates_name], axis=0),
        H.make_node("Concat", [batch_name, black_name, coords_name], [black_idx_name], axis=1),
        H.make_node("Concat", [batch_name, green_name, coords_name], [green_idx_name], axis=1),
        H.make_node("Concat", [black_idx_name, green_idx_name], [idx_u8_name], axis=0),
        H.make_node("Cast", [idx_u8_name], [idx_i64_name], to=TP.INT64),
        H.make_node("ScatterND", [input_name, idx_i64_name, updates_name], [output_name],
                    reduction="add"),
    ]
    _tag(nodes, tag)
    inits = [
        (batch_name, np.zeros((10, 1), dtype=np.uint8)),
        (black_name, np.zeros((10, 1), dtype=np.uint8)),
        (green_name, np.full((10, 1), color, dtype=np.uint8)),
    ]
    return nodes, inits, [1, 10, 30, 30]


def task224_status_tail(input_name="input", row_code_name="row_code16_u8",
                        col_code_name="col_code16_u8", paint_name="paint_f32",
                        output_name="output", *, crop=(0, 0, 16, 16),
                        outside_col_code=3, row_t_name="task224_row_col_u8",
                        mask16_name="task224_mask16_b",
                        mask30_name="task224_mask30_b",
                        pads_name="task224_mask_pads"):
    """Terminal renderer for task224's status-code frame.

    Executable lowering: compare row/column status codes on the certified
    16x16 crop, pad the bool selector false outside the crop, then paint from
    `paint` or pass through `input`. All intermediate outputs are charged normally.
    """
    if tuple(int(v) for v in crop) != (0, 0, 16, 16):
        raise ValueError("task224_status_tail only supports crop=(0,0,16,16)")
    if int(outside_col_code) != 3:
        raise ValueError("task224_status_tail expects outside_col_code=3")
    tag = f"{TASK224_STATUS_DOC};crop=0,0,16,16;outside_col_code=3"
    nodes = [
        H.make_node("Transpose", [row_code_name], [row_t_name], perm=[1, 0]),
        H.make_node("Greater", [row_t_name, col_code_name], [mask16_name]),
        H.make_node("Pad", [mask16_name, pads_name], [mask30_name], mode="constant"),
        H.make_node("Where", [mask30_name, paint_name, input_name], [output_name]),
    ]
    _tag(nodes, tag)
    inits = [
        (pads_name, np.array([0, 0, 14, 14], dtype=np.int64)),
    ]
    return nodes, inits, [1, 10, 30, 30]


def terminal_hodel_ray(input_name="input", x_name="x_u8", row_eq_name="row_eq_b",
                       row_side_name="row_side_b", col_eq_name="col_eq_b",
                       col_side_name="col_side_b", marker_oh_name="marker_oh_b",
                       bg_oh_name="bg_oh_b", hflag_name="hflag_b",
                       vflag_name="vflag_b", output_name="output", *,
                       x_bg_name="term_x_bg_b", hline_name="term_hline_b",
                       hray_name="term_hray_b", vline_name="term_vline_b",
                       vray_name="term_vray_b", ray_name="term_ray_b",
                       paint_name="term_paint_b", not_bg_name="term_not_bg_b",
                       marker_only_name="term_marker_only_b",
                       marker_f_name="term_marker_f"):
    """Terminal renderer for task051's factored Hodel ray.

    Executable lowering: build the ray from row/column equality and side factors,
    intersect it with the relowered background channel, then terminal-Where marker
    one-hot over the original input.
    ⚠️ COST (fixed 2026-07-10): EVERY intermediate in this chain is charged by the real
    Kaggle grader — the "tagged chain is skipped" claim was a local scorer hack (#p807
    proved it, e.g. task206 19.66→12.36 true). save() auto-charges all tmps; price the
    whole chain before choosing this renderer.
    """
    tag = TERMINAL_HODEL_RAY_DOC
    nodes = [
        H.make_node("Cast", [x_name], [x_bg_name], to=TP.BOOL),
        H.make_node("And", [row_eq_name, col_side_name], [hline_name]),
        H.make_node("And", [hline_name, hflag_name], [hray_name]),
        H.make_node("And", [col_eq_name, row_side_name], [vline_name]),
        H.make_node("And", [vline_name, vflag_name], [vray_name]),
        H.make_node("Or", [hray_name, vray_name], [ray_name]),
        H.make_node("And", [ray_name, x_bg_name], [paint_name]),
        H.make_node("Not", [bg_oh_name], [not_bg_name]),
        H.make_node("And", [marker_oh_name, not_bg_name], [marker_only_name]),
        H.make_node("Cast", [marker_only_name], [marker_f_name], to=TP.FLOAT),
        H.make_node("Where", [paint_name, marker_f_name, input_name], [output_name]),
    ]
    _tag(nodes, tag)
    return nodes, [], [1, 10, 30, 30]


def _tag(nodes, doc):
    for n in nodes:
        n.doc_string = doc
    return nodes


def input_u8(input_name="input", output_name="grid_u8", *, valid_name="valid_b",
             cast_name="input_u8_cast", weighted_name="input_weighted_u8",
             raw_name="grid_raw_u8", colors_name="colors10_u8",
             sentinel_name="sentinel255_u8"):
    """Executable one-hot -> uint8 color-id plane with 255 outside-grid sentinel.

    The graph is standard ONNX: Cast, Mul, ReduceMax, Where. Every intermediate is charged.
    """
    colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    nodes = [
        H.make_node("ReduceMax", [input_name], [valid_name], axes=[1], keepdims=1),
        H.make_node("Cast", [valid_name], [valid_name + "_b"], to=TP.BOOL),
        H.make_node("Cast", [input_name], [cast_name], to=TP.UINT8),
        H.make_node("Mul", [cast_name, colors_name], [weighted_name]),
        H.make_node("ReduceMax", [weighted_name], [raw_name], axes=[1], keepdims=1),
        H.make_node("Where", [valid_name + "_b", raw_name, sentinel_name], [output_name]),
    ]
    _tag(nodes, f"{INPUT_U8_DOC};scratch=1")
    inits = [
        (colors_name, colors),
        (sentinel_name, np.array(255, dtype=np.uint8)),
    ]
    return nodes, inits, [1, 1, 30, 30]


def compact_bbox_u8(grid_name="grid_u8", presence_name="presence_u8", bbox_name="bbox_u8",
                    *, mask_name="mask10_b", mask_f_name="mask10_f",
                    counts_name="counts_f", colors_name="colors10_u8",
                    axes_hw_name="axes_hw", four_name="four_f"):
    """Tagged per-color presence and bbox helper.

    Emits executable standard ONNX for color masks, counts, row/column profiles and
    ArgMax min/max.
    ⚠️ COST (fixed 2026-07-10): ALL outputs are charged by the real Kaggle grader —
    masks, counts, and row/column profiles included; "tagged scratch" was a local scorer
    hack and is not honored on Kaggle. save() auto-charges everything.
    """
    nodes = [
        H.make_node("Equal", [grid_name, colors_name], [mask_name]),
        H.make_node("Cast", [mask_name], [mask_f_name], to=TP.FLOAT),
        H.make_node("ReduceSum", [mask_f_name, axes_hw_name], [counts_name], keepdims=0),
        H.make_node("Equal", [counts_name, four_name], ["presence_b"]),
        H.make_node("Cast", ["presence_b"], [presence_name], to=TP.UINT8),
        H.make_node("ReduceMax", [mask_f_name], ["row_prof_f"], axes=[3], keepdims=0),
        H.make_node("ReduceMax", [mask_f_name], ["col_prof_f"], axes=[2], keepdims=0),
        H.make_node("ArgMax", ["row_prof_f"], ["rmin_i64"], axis=2, keepdims=0),
        H.make_node("ArgMax", ["row_prof_f"], ["rmax_i64"], axis=2, keepdims=0, select_last_index=1),
        H.make_node("ArgMax", ["col_prof_f"], ["cmin_i64"], axis=2, keepdims=0),
        H.make_node("ArgMax", ["col_prof_f"], ["cmax_i64"], axis=2, keepdims=0, select_last_index=1),
        H.make_node("Cast", ["rmin_i64"], ["rmin_u8"], to=TP.UINT8),
        H.make_node("Cast", ["rmax_i64"], ["rmax_u8"], to=TP.UINT8),
        H.make_node("Cast", ["cmin_i64"], ["cmin_u8"], to=TP.UINT8),
        H.make_node("Cast", ["cmax_i64"], ["cmax_u8"], to=TP.UINT8),
        H.make_node("Unsqueeze", ["rmin_u8", "axis2_i64"], ["rmin_3_u8"]),
        H.make_node("Unsqueeze", ["rmax_u8", "axis2_i64"], ["rmax_3_u8"]),
        H.make_node("Unsqueeze", ["cmin_u8", "axis2_i64"], ["cmin_3_u8"]),
        H.make_node("Unsqueeze", ["cmax_u8", "axis2_i64"], ["cmax_3_u8"]),
        H.make_node("Concat", ["rmin_3_u8", "rmax_3_u8", "cmin_3_u8", "cmax_3_u8"], [bbox_name], axis=2),
    ]
    for n in nodes:
        if n.output and n.output[0] not in {presence_name, bbox_name}:
            n.doc_string = f"{COMPACT_BBOX_U8_DOC};scratch=1"
    inits = [
        (colors_name, np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
        (axes_hw_name, np.array([2, 3], dtype=np.int64)),
        (four_name, np.array(4.0, dtype=np.float32)),
        ("axis2_i64", np.array([2], dtype=np.int64)),
    ]
    return nodes, inits, [1, 10], [1, 10, 4]


def gather_nd_u8(mask_name="mask10_b", bbox_name="bbox_u8", sig_name="sig9_b"):
    """Tagged 3x3 bbox endpoint/midpoint sampler producing [1,10,9] bool."""
    nodes = [
        H.make_node("Cast", [bbox_name], ["bbox_i64"], to=TP.INT64),
        H.make_node("Gather", ["bbox_i64", "idx0_i64"], ["rmin_g"], axis=2),
        H.make_node("Gather", ["bbox_i64", "idx1_i64"], ["rmax_g"], axis=2),
        H.make_node("Gather", ["bbox_i64", "idx2_i64"], ["cmin_g"], axis=2),
        H.make_node("Gather", ["bbox_i64", "idx3_i64"], ["cmax_g"], axis=2),
        H.make_node("Add", ["rmin_g", "rmax_g"], ["rsum_g"]),
        H.make_node("Add", ["cmin_g", "cmax_g"], ["csum_g"]),
        H.make_node("Div", ["rsum_g", "two_i64"], ["rmid_g"]),
        H.make_node("Div", ["csum_g", "two_i64"], ["cmid_g"]),
        H.make_node("Concat", ["rmin_g", "rmid_g", "rmax_g"], ["rows3_i64"], axis=2),
        H.make_node("Concat", ["cmin_g", "cmid_g", "cmax_g"], ["cols3_i64"], axis=2),
        H.make_node("Gather", ["rows3_i64", "rowpat_i64"], ["rows9_i64"], axis=2),
        H.make_node("Gather", ["cols3_i64", "colpat_i64"], ["cols9_i64"], axis=2),
        H.make_node("Unsqueeze", ["rows9_i64", "axis3_i64"], ["rows9_4_i64"]),
        H.make_node("Unsqueeze", ["cols9_i64", "axis3_i64"], ["cols9_4_i64"]),
        H.make_node("Concat", ["rows9_4_i64", "cols9_4_i64"], ["gather_idx_i64"], axis=3),
        H.make_node("GatherND", [mask_name, "gather_idx_i64"], [sig_name], batch_dims=2),
    ]
    for n in nodes:
        if n.output and n.output[0] != sig_name:
            n.doc_string = f"{GATHER_ND_U8_DOC};scratch=1"
    inits = [
        ("idx0_i64", np.array([0], dtype=np.int64)),
        ("idx1_i64", np.array([1], dtype=np.int64)),
        ("idx2_i64", np.array([2], dtype=np.int64)),
        ("idx3_i64", np.array([3], dtype=np.int64)),
        ("two_i64", np.array(2, dtype=np.int64)),
        ("rowpat_i64", np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)),
        ("colpat_i64", np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)),
        ("axis3_i64", np.array([3], dtype=np.int64)),
    ]
    return nodes, inits, [1, 10, 9]


def scatter_nd_u8(colors_name="slot_colors", coords_name="slot_coords",
                  bg_name="bg_u8", output_name="state7_u8"):
    """Tagged symmetric-fragment scatter helper.

    `coords_name` stores fragment descriptors `(row_offset, col_offset)` for up to four
    colors. Row offset zero denotes the cross fragment; otherwise the helper scatters
    the four reflected rectangle corners into a 7x7 color-id state.
    """
    # The full lowering is task-specific enough that build.py owns it; this helper only
    # documents the public surface. There is no scorer-recognized scratch exemption.
    return [], [], [1, 1, 7, 7]


class T:
    def __init__(self, name, shape, dt): self.name, self.shape, self.dt = name, list(shape), dt
    def __repr__(self): return f"T({self.name},{self.shape},{self.dt})"
    @property
    def nbytes(self): return int(np.prod(self.shape)) * DT[self.dt][1]


class G:
    def __init__(self, task=None):
        self.task = task
        self.nodes, self.inits, self.charged = [], [], []
        self.params = 0
        self._n = 0
        self.opset = 17
        self.done = False
        self.input = T("input", [1, 10, 30, 30], "f32")

    # ---------- internals ----------
    def _nm(self): self._n += 1; return f"t{self._n}"

    def init(self, arr, dt=None):
        """Register an initializer (charged as PARAMS = element count, dtype-free)."""
        arr = np.asarray(arr)
        if dt: arr = arr.astype(DT[dt][2])
        dtk = NP2DT[arr.dtype.type]
        name = self._nm()
        self.inits.append(H.make_tensor(name, DT[dtk][0], arr.shape, arr.ravel()))
        self.params += int(arr.size)
        return T(name, arr.shape, dtk)

    def n(self, op, ins, shape, dt, is_output=False, **attrs):
        """Generic node. Output charged as MEMORY unless it is the graph output."""
        out = "output" if is_output else self._nm()
        self.nodes.append(H.make_node(op, [i.name if isinstance(i, T) else i for i in ins],
                                      [out], **attrs))
        t = T(out, shape, dt)
        if is_output: self.done, self.out_t = True, t
        else: self.charged.append(t)
        if op.startswith("Bitwise") or op == "BitShift": self.opset = max(self.opset, 18)
        return t

    # ---------- input access (input tensor itself is FREE) ----------
    def slice(self, x, starts, ends, axes, steps=None):
        ins = [x, self.init(starts, "i64"), self.init(ends, "i64"), self.init(axes, "i64")]
        if steps: ins.append(self.init(steps, "i64"))
        sh = list(x.shape)
        for k, ax in enumerate(axes):
            st = (steps or [1]*len(axes))[k]
            lo, hi = starts[k], min(ends[k], x.shape[ax]) if ends[k] >= 0 else x.shape[ax]+ends[k]
            if lo < 0: lo += x.shape[ax]
            sh[ax] = max(0, -(-abs(hi - lo) // abs(st)))
        return self.n("Slice", ins, sh, x.dt)

    def color_plane(self, c, cast=None, h=30, w=30):
        """One-hot input channel c == binary mask of color c. Optionally cast (u8/b)."""
        p = self.slice(self.input, [c, 0, 0], [c+1, h, w], [1, 2, 3])
        return self.cast(p, cast) if cast else p

    def relower_onehot_plane(self, channel=0, crop=None, dtype="b"):
        nodes, inits, shape = relower_onehot_plane(
            self.input.name, self._nm(), channel=channel, crop=crop, dtype=dtype,
            starts_name=self._nm(), ends_name=self._nm())
        for name, arr in inits:
            arr = np.asarray(arr, np.int64)
            self.inits.append(H.make_tensor(name, TP.INT64, arr.shape, arr.ravel()))
            self.params += int(arr.size)
        self.nodes.extend(nodes)
        t = T(nodes[-1].output[0], shape, dtype)
        self.charged.append(t)
        return t

    def crop(self, x, h0, h1, w0, w1):
        return self.slice(x, [h0, w0], [h1, w1], [len(x.shape)-2, len(x.shape)-1])

    # ---------- elementwise ----------
    def cast(self, x, dt): return self.n("Cast", [x], x.shape, dt, to=DT[dt][0])
    def _bshape(self, a, b):
        sa = a.shape if isinstance(a, T) else [1]; sb = b.shape if isinstance(b, T) else [1]
        L = max(len(sa), len(sb)); sa = [1]*(L-len(sa))+list(sa); sb = [1]*(L-len(sb))+list(sb)
        return [max(x, y) for x, y in zip(sa, sb)]
    def _bin(self, op, a, b, dt=None):
        if not isinstance(b, T): b = self.init(np.array(b, DT[a.dt][2]))
        return self.n(op, [a, b], self._bshape(a, b), dt or a.dt)
    def add(self, a, b): return self._bin("Add", a, b)
    def sub(self, a, b): return self._bin("Sub", a, b)
    def mul(self, a, b): return self._bin("Mul", a, b)
    def eq(self, a, b): return self._bin("Equal", a, b, "b")
    def greater(self, a, b): return self._bin("Greater", a, b, "b")
    def less(self, a, b): return self._bin("Less", a, b, "b")
    def and_(self, a, b): return self._bin("And", a, b, "b")
    def or_(self, a, b): return self._bin("Or", a, b, "b")
    def not_(self, x): return self.n("Not", [x], x.shape, "b")
    def where(self, c, a, b):
        a_t, b_t = isinstance(a, T), isinstance(b, T)
        if not a_t: a = self.init(np.array(a))
        if not b_t: b = self.init(np.array(b))
        # broadcast ALL THREE (cond + both branches). The old code coerced a,b to T first, so
        # `a if isinstance(a,T) else b` was always `a` → the else-branch `b` was dropped from
        # the shape, undersizing memory up to 10x (audit 2026-07-08 / T7). 3-way _bshape can
        # only INCREASE the shape, so it can never introduce a new undercount. dtype comes
        # from the real tensor branch when the other side is a coerced scalar.
        dt = a.dt if (a_t or not b_t) else b.dt
        return self.n("Where", [c, a, b], self._bshape(self._bshape(c, a), b), dt)
    def clip(self, x, lo, hi):
        return self.n("Clip", [x, self.init(np.array(lo, DT[x.dt][2])),
                               self.init(np.array(hi, DT[x.dt][2]))], x.shape, x.dt)

    # ---------- structure ----------
    def reshape(self, x, shape):
        return self.n("Reshape", [x, self.init(shape, "i64")], shape, x.dt)
    def transpose(self, x, perm):
        return self.n("Transpose", [x], [x.shape[p] for p in perm], x.dt, perm=perm)
    def concat(self, ts, axis):
        sh = list(ts[0].shape); sh[axis] = sum(t.shape[axis] for t in ts)
        return self.n("Concat", ts, sh, ts[0].dt, axis=axis)
    def pad_to(self, x, h, w, value=0):
        """Zero-pad last two dims to h×w (bottom/right)."""
        r = len(x.shape)
        pads = [0]*r + [0]*r
        pads[r-2+r], pads[r-1+r] = h - x.shape[-2], w - x.shape[-1]
        sh = list(x.shape); sh[-2], sh[-1] = h, w
        ins = [x, self.init(pads, "i64")]
        if value: ins.append(self.init(np.array(value, DT[x.dt][2])))
        return self.n("Pad", ins, sh, x.dt)

    # ---------- spatial ----------
    def maxpool(self, x, k, pads=None):
        p = pads if pads is not None else (k - 1) // 2
        return self.n("MaxPool", [x], x.shape, x.dt, kernel_shape=[k, k],
                      pads=[p, p, p, p])
    def conv_int(self, x, w, pads=0):
        """ConvInteger: uint8 activations × int8 kernel → int32. w = numpy array [O,I,kh,kw]."""
        w = np.asarray(w, np.int8)
        wt = self.init(w)
        p = pads if isinstance(pads, (list, tuple)) else [pads]*4
        oh = x.shape[-2] + p[0] + p[2] - w.shape[2] + 1
        ow = x.shape[-1] + p[1] + p[3] - w.shape[3] + 1
        return self.n("ConvInteger", [x, wt], [1, w.shape[0], oh, ow], "i32", pads=list(p))
    def reduce(self, x, op, axes, keepdims=1):
        ax = [a % len(x.shape) for a in axes]
        sh = [(1 if i in ax else d) for i, d in enumerate(x.shape)]
        if not keepdims: sh = [d for i, d in enumerate(sh) if i not in ax]
        # ReduceSum moved axes to an INPUT at opset 13 (the other Reduce* ops at 18), so
        # axes-as-attribute is opset-illegal for Sum and check_model rejects it (T7). Pass an
        # int64 axes initializer for Sum; keep axes-as-attribute for the rest at opset 17.
        if op == "Sum":
            axt = self.init(np.array(ax, np.int64))
            return self.n("ReduceSum", [x, axt], sh, x.dt, keepdims=keepdims)
        return self.n(f"Reduce{op}", [x], sh, x.dt, axes=ax, keepdims=keepdims)
    def argmax(self, x, axis, keepdims=1):
        sh = list(x.shape); sh[axis] = 1
        if not keepdims: sh.pop(axis)
        return self.n("ArgMax", [x], sh, "i64", axis=axis, keepdims=keepdims)
    def einsum(self, eq, ts, out_shape, out_dt=None):
        return self.n("Einsum", ts, out_shape, out_dt or ts[0].dt, equation=eq)

    # ---------- tables / indexing ----------
    def gather(self, table, idx, axis=0, clamp=True):
        """Gather with AUTO-CLAMP of computed indices (OOB = crash = FAIL)."""
        if isinstance(table, np.ndarray): table = self.init(table)
        if isinstance(idx, T) and clamp:
            hi = table.shape[axis] - 1
            idx = self.clip(idx, np.array(0, DT[idx.dt][2]), np.array(hi, DT[idx.dt][2]))
        elif not isinstance(idx, T):
            idx = self.init(np.asarray(idx, np.int64))
        sh = list(table.shape[:axis]) + list(idx.shape) + list(table.shape[axis+1:])
        return self.n("Gather", [table, idx], sh, table.dt if isinstance(table, T) else "u8",
                      axis=axis)
    def channel_permute(self, mapping):
        """RECOLORING for free-ish: output[c] = input[mapping[c]]. 10 params, one op."""
        return self.n("Gather", [self.input, self.init(mapping, "i64")],
                      [1, 10, 30, 30], "f32", axis=1)

    def flip(self, x, axis):
        """Mirror along axis (hmirror: axis=-2, vmirror: axis=-1). Safe negative-step Slice."""
        ax = axis % len(x.shape)
        n = x.shape[ax]
        return self.n("Slice", [x, self.init([n - 1], "i64"), self.init([-n - 1], "i64"),
                                self.init([ax], "i64"), self.init([-1], "i64")],
                      x.shape, x.dt)
    def tile(self, x, reps):
        return self.n("Tile", [x, self.init(reps, "i64")],
                      [d * r for d, r in zip(x.shape, reps)], x.dt)
    def upscale(self, x, k):
        """Nearest-neighbor upscale of the last two dims by k (Resize is a legal op)."""
        sh = list(x.shape); sh[-2] *= k; sh[-1] *= k
        scales = self.init(np.array([1.0] * (len(x.shape) - 2) + [k, k], np.float32))
        return self.n("Resize", [x, "", scales.name], sh, x.dt,
                      mode="nearest", nearest_mode="floor")
    def min_(self, a, b): return self._bin("Min", a, b)
    def max_(self, a, b): return self._bin("Max", a, b)

    # ---------- composite idioms ----------
    def flood(self, seed, mask, rounds):
        """Geodesic flood: rounds × (dilate ∧ mask). Cost = rounds × 2 planes. Cap rounds
        with the CERTIFIED diameter, not a guess."""
        s = seed
        for _ in range(rounds):
            s = self.mul(self.maxpool(s, 3), mask)
        return s
    def bbox_mask(self, m):
        """Bounding-box mask of a [1,1,H,W] u8 mask via 4 tiny vectors (no big planes)."""
        H_, W_ = m.shape[-2], m.shape[-1]
        rows = self.reduce(m, "Max", [-1])                       # [1,1,H,1]
        cols = self.reduce(m, "Max", [-2])                       # [1,1,1,W]
        rcum1 = self.n("MaxPool", [rows], rows.shape, rows.dt,   # rows at/after first hit
                       kernel_shape=[H_, 1], pads=[H_-1, 0, 0, 0])
        rcum2 = self.n("MaxPool", [rows], rows.shape, rows.dt,   # rows at/before last hit
                       kernel_shape=[H_, 1], pads=[0, 0, H_-1, 0])
        ccum1 = self.n("MaxPool", [cols], cols.shape, cols.dt,
                       kernel_shape=[1, W_], pads=[0, W_-1, 0, 0])
        ccum2 = self.n("MaxPool", [cols], cols.shape, cols.dt,
                       kernel_shape=[1, W_], pads=[0, 0, 0, W_-1])
        r = self.mul(rcum1, rcum2)                               # [1,1,H,1] in-box rows
        c = self.mul(ccum1, ccum2)                               # [1,1,1,W] in-box cols
        return self.mul(r, c)                                    # broadcast → [1,1,H,W]

    # ---------- terminal renderers (output tensor is FREE) ----------
    def out_equal_arange(self, state, K=10):
        """state: [1,1,30,30] integer color-codes → output [1,10,30,30] bool. ONE free op.
        (OneHot is NOT_IMPLEMENTED in ORT 1.24 — this is the replacement.)"""
        ar = self.init(np.arange(K).reshape(1, K, 1, 1), state.dt)
        return self.n("Equal", [state, ar], [1, 10, 30, 30], "b", is_output=True)
    def out_cast(self, x, dt="f32"):
        """Final op = Cast producing the (free) output. x must be [1,10,30,30]."""
        return self.n("Cast", [x], x.shape, dt, to=DT[dt][0], is_output=True)
    def out_identity(self, x):
        return self.n("Identity", [x], x.shape, x.dt, is_output=True)

    def task224_status_tail(self, row_code16, col_code16, paint, *,
                            crop=(0, 0, 16, 16), outside_col_code=3):
        nodes, inits, shape = task224_status_tail(
            self.input.name, row_code16.name, col_code16.name, paint.name, "output",
            crop=crop, outside_col_code=outside_col_code,
            row_t_name=self._nm(), mask16_name=self._nm(),
            mask30_name=self._nm(), pads_name=self._nm())
        for name, arr in inits:
            arr = np.asarray(arr, np.int64)
            self.inits.append(H.make_tensor(name, TP.INT64, arr.shape, arr.ravel()))
            self.params += int(arr.size)
        self.nodes.extend(nodes)
        self.done, self.out_t = True, T("output", shape, "f32")
        return self.out_t

    def terminal_hodel_ray(self, x, row_eq, row_side, col_eq, col_side, marker_oh,
                           bg_oh, hflag, vflag):
        nodes, inits, shape = terminal_hodel_ray(
            self.input.name, x.name, row_eq.name, row_side.name, col_eq.name,
            col_side.name, marker_oh.name, bg_oh.name, hflag.name, vflag.name,
            "output",
            x_bg_name=self._nm(), hline_name=self._nm(), hray_name=self._nm(),
            vline_name=self._nm(), vray_name=self._nm(), ray_name=self._nm(),
            paint_name=self._nm(), not_bg_name=self._nm(),
            marker_only_name=self._nm(), marker_f_name=self._nm())
        for name, arr in inits:
            arr = np.asarray(arr)
            dtk = NP2DT[arr.dtype.type]
            self.inits.append(H.make_tensor(name, DT[dtk][0], arr.shape, arr.ravel()))
            self.params += int(arr.size)
        self.nodes.extend(nodes)
        self.done, self.out_t = True, T("output", shape, "f32")
        return self.out_t

    # ---------- accounting & save ----------
    def cost(self): return self.params + sum(t.nbytes for t in self.charged)
    def points(self): return max(1.0, 25.0 - math.log(max(1, self.cost())))
    def budget(self):
        rows = sorted(self.charged, key=lambda t: -t.nbytes)
        s = [f"BUDGET: params={self.params}  memory={sum(t.nbytes for t in self.charged)}"
             f"  cost={self.cost()}  → {self.points():.4f} pts"]
        for t in rows[:12]:
            s.append(f"  {t.nbytes:>6} B  {t.dt:<4} {t.shape}")
        return "\n".join(s)
    def save(self, path, ort_check=True):
        assert self.done, "no output yet — finish with an out_*() renderer"
        graph = H.make_graph(
            self.nodes, "g",
            [H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
            [H.make_tensor_value_info("output", DT[self.out_t.dt][0], self.out_t.shape)],
            initializer=self.inits)
        m = H.make_model(graph, opset_imports=[H.make_opsetid("", self.opset)])
        m.ir_version = 8
        onnx.checker.check_model(m, full_check=True)
        g = onnx.shape_inference.infer_shapes(m, strict_mode=True)
        # verify the tracked budget against the grader's static accounting.
        # 2026-07-10: the old "tagged scratch" exemptions (relower/task224/hodel-ray tmps)
        # are GONE — those skips only ever existed in a locally-patched neurogolf_utils;
        # Kaggle probes #p807..GT6 proved the real grader charges EVERY node output except
        # the graph output. doc_string tags are cosmetic; no tensor is free.
        vi = {t.name: t for t in list(g.graph.value_info) + list(g.graph.output)}
        real, real_by_name = 0, {}
        for node in g.graph.node:
            for o in node.output:
                if not o or o == "output" or o not in vi:
                    continue
                tt = vi[o].type.tensor_type
                dims = [d.dim_value for d in tt.shape.dim]
                b = int(np.prod(dims)) * np.dtype(
                    onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)).itemsize
                real += b
                real_by_name[o] = b
        # Per-tensor undercount check on tracked tensors (audit 2026-07-08 / T7): an
        # undersized where()/shape estimate is a genuine tracking bug — FAIL LOUD.
        for t in self.charged:
            rb = real_by_name.get(t.name)
            if rb is not None and rb > t.nbytes:
                raise AssertionError(
                    f"COST UNDERCOUNT on {t.name}: tracked {t.nbytes} B < shape-inferred "
                    f"{rb} B — a where()/shape estimate is undersized; fix the tracking.")
        # Auto-charge grader-counted tensors the builder didn't track (tmps inside tagged
        # composite primitives — the grader charges them; the old skip was a local hack).
        tracked_names = {t.name for t in self.charged}
        for name, b in real_by_name.items():
            if name not in tracked_names and b > 0:
                self.charged.append(T(name, [b], "u8"))  # exact bytes, dtype-agnostic
                print(f"AUTO-CHARGED {b} B for {name} (grader counts it; 'tagged scratch' "
                      f"is NOT free on Kaggle — scorer hack fixed 2026-07-10)")
        tracked = sum(t.nbytes for t in self.charged)
        if real < tracked:
            print(f"WARNING: tracked memory {tracked} > shape-inferred {real} "
                  f"(over-count — safe direction, but a shape estimate is loose; trust {real})")
        if ort_check:
            import onnxruntime as ort
            o = ort.SessionOptions(); o.log_severity_level = 4
            s = ort.InferenceSession(m.SerializeToString(), o)
            d = np.zeros((1, 10, 30, 30), np.float32); d[0, 0, 0, 0] = 1
            s.run(None, {"input": d})
        onnx.save(m, path)
        print(self.budget())
        print(f"saved {path} (opset {self.opset}) — now run fast_verify")
        return path
