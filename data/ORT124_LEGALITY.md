# ORT-1.24 measured legality sheet (fleet evidence, not spec — analyst-distilled, entries cite the tasks that hit them)

## Dead on ORT 1.24 (measured)
`Einsum (uint8/int8/integer) — NOT_IMPLEMENTED / no path / UNSCORABLE — (tasks: 005,009,017)`
`Scan — in EXCLUDED_OPS / "scan_fused is impossible under this gate" — (tasks: 004,009)`
`bool Where — NOT_IMPLEMENTED — (tasks: 002,014)`
`Pad (bool/broadcast) — ORT Pad/Max broadcast failure — (tasks: 002,008)`
`Gather (uint8 indices) — InferenceError: unsupported type: tensor(uint8) — (tasks: 005) (single report)`
`ConvTranspose (uint8/int8/int32) — ONNX shape inference rejects — (tasks: 009) (single report)`
`ConvInteger (float one-hot) — cannot consume float one-hot input — (tasks: 008) (single report)`
`OneHot (bool) — unsupported → dropped for Equal — (tasks: 004) (single report)`

## Legal but trapped (measured)
`int64 ScatterND — legal but high cost (indices) — (tasks: 005,008)`
`fp32 Conv before uint8 Cast — scorer charges full float32 outputs — (tasks: 008) (single report)`
`DepthToSpace (u8) — legal u8 renderer but exceeds bar — (tasks: 009) (single report)`
`MaxPool/tile — legal but high cost floors — (tasks: 009) (single report)`
`one-hot + channel-0 expansion — must pay full in ONNX — (tasks: 005) (single report)`
`i32 Cast for indices — +3600 B for 30×30 — (tasks: 005) (single report)`
`fp16 Cast on input — +18000 B for legal fp16 path — (tasks: 005) (single report)`
`ConvInteger (int32 outputs) — two [1,8,7,7] = 3136 B before reduction — (tasks: 005) (single report)`
`float ConvTranspose — float-only forces costly f16 10ch state — (tasks: 009) (single report)`
`Equal (terminal crop/pad) — emits cropped shape vs [1,10,30,30] gate — (tasks: 008) (single report)`
