# Neurons Volume Visualizer — Implementation Guide

A web-based neuroglancer-style volume viewer with 4-panel layout (axial, coronal, sagittal, 3D Gaussian splats), built with FastAPI + vanilla JS + Three.js.

## Architecture

```
neurons/visualizer/
  __init__.py              # package marker
  __main__.py              # CLI: python -m neurons.visualizer --raw X --seg Y
  volume_loader.py         # unified loader for HDF5/TIFF/NRRD/NIfTI/NumPy
  app.py                   # FastAPI REST endpoints + PNG composition
  static/
    index.html             # single-page shell, 2×2 grid + toolbar
    style.css              # dark theme, neuroglancer-style
    app.js                 # 2D slice rendering, crosshairs, selection, boot
    volume_renderer.js     # Three.js 3D panel: splats + slice planes
```

```
┌──────────────────────────────────────────────-───┐
│                  Browser (SPA)                   │
│ ┌───────────┬───────────┐                        │
│ │ Axial XY  │ Sagittal  │  app.js: slice extract │
│ │ (canvas)  │ ZY(canvas)│  + overlay composite   │
│ ├───────────┼───────────┤                        │
│ │ Coronal XZ│ 3D Splats │  volume_renderer.js:   │
│ │ (canvas)  │ (WebGL2)  │  Three.js + GLSL       │
│ └───────────┴───────────┘                        │
│ [ Overlay ] [ Opacity ] [ Colormap ] [ Instance ]│
└──────────────────────┬──────────────────────────-┘
                       │ fetch /api/…
              ┌────────▼────────┐
              │   FastAPI app   │
              │   (app.py)      │
              │ ┌──────────────┐│
              │ │ VolumeLoader ││
              │ │ (HDF5/TIFF…) ││
              │ └──────────────┘│
              └─────────────────┘
```

## Step-by-Step Implementation

### Step 1 — Volume Loader (`volume_loader.py`)

**Goal**: Load any supported volumetric format into a unified `VolumeData(data, spacing)`.

- `VolumeData` wraps a 3D numpy array (Z, Y, X order) with voxel spacing `(sz, sy, sx)`.
- `load_volume(path, key, spacing)` auto-detects format by extension and delegates to existing preprocessors.
- Handles squeeze (4D→3D), spacing extraction from HDF5 attrs / NRRD headers, numpy files.
- Provides `slice_axis(axis, idx)` for 2D slices and `to_uint8()` for display normalisation.

**Key decision**: spacing is `(sz, sy, sx)` in ZYX order, matching the data array layout. This is essential for correct anisotropic rendering.

### Step 2 — FastAPI Backend (`app.py`)

**Goal**: Serve volume data via REST endpoints for the browser frontend.

**Endpoints** (all under `/api/volumes/{vid}/`):

| Endpoint | Returns | Purpose |
|---|---|---|
| `POST /api/volumes` | JSON `{id, shape, dtype, spacing}` | Register volumes |
| `GET  .../meta` | JSON metadata | Shape, spacing, has_seg |
| `GET  .../slice/{axis}/{idx}` | PNG image | Server-side slice + overlay composite |
| `GET  .../chunk?downsample=N` | Raw uint8 bytes | Downsampled raw for 3D textures |
| `GET  .../seg_chunk?downsample=N` | RGBA bytes | Palette-coloured seg for 3D textures |
| `GET  .../seg_id_chunk?downsample=N` | uint8 bytes | Seg IDs mod 256 for shader selection |
| `GET  .../labels` | JSON label list + palette | Unique IDs + RGB colours |

**Key decisions**:
- The `chunk` and `seg_chunk` endpoints do the heavy lifting of downsampling server-side, so the browser receives manageable arrays (~256³).
- Seg IDs are reduced to uint8 (mod 256) for GPU texture lookup — sufficient for up to 255 distinct segment colours.
- A random RGB palette is generated per-session with seed=0 for reproducibility.
- The slice endpoint composites raw grayscale + seg overlay using alpha blending, returning PNG.

**In-memory store**: `_store[vid] = _VolumeEntry(raw, seg, palette)`. The CLI (`__main__.py`) pre-loads volumes with `vid="default"`.

### Step 3 — HTML Shell + CSS (`index.html`, `style.css`)

**Goal**: 2×2 panel grid with a toolbar.

```
┌──────────────┬──────────────┐
│   Axial (XY) │ Sagittal(YZ) │
├──────────────┼──────────────┤
│ Coronal (XZ) │  3D Volume   │
└──────────────┴──────────────┘
[ Overlay ✓ ] [ Opacity ━━━ ] [ Colormap ▾ ] [ Instance: #N ] [ Z:Y:X ]
```

- CSS Grid 2×2 layout, each cell contains a `<canvas>`.
- Dark theme (#1a1a2e background), neuroglancer-inspired.
- Toolbar with checkbox, range slider, select, info spans.
- Selection chips: inline pills with × to deselect, "clear" link.
- Three.js loaded from CDN (`three@0.158.0`).

### Step 4 — Client-Side Slice Rendering (`app.js`)

**Goal**: Instant slice browsing with no server round-trips.

#### 4.1 Boot sequence

1. Fetch `/api/volumes/default/meta` → shape, spacing, has_seg.
2. Compute downsample factor: `ds = ceil(maxXY / 512)`, cap at 40M voxels.
3. Fetch raw chunk, seg RGBA chunk, seg ID chunk (all cached via Browser Cache API).
4. Store as `window.NV.rawVol`, `segVol`, `segIdVol` (typed arrays in ZYX C-order).
5. Render all panels, start 3D renderer.

#### 4.2 Slice extraction

`extractSlice(axis)` extracts a 2D RGBA `ImageData` directly from the preloaded typed arrays:
- **Axial (axis=0)**: rows=Y, cols=X, indexed at `z * dY*dX + y*dX + x`
- **Coronal (axis=1)**: rows=Z, cols=X, indexed at `z * dY*dX + y_slice*dX + x`
- **Sagittal (axis=2)**: rows=Y, cols=Z (transposed for co-axis alignment), indexed at `z * dY*dX + y*dX + x_slice`

Overlay compositing: per-pixel alpha blend `(1-a)*gray + a*segColor`. Selected segments get full opacity, unselected get `opacity * 0.15`.

#### 4.3 Synchronized zoom & pan

All three panels share a single `globalView = { zoom, panPixX, panPixY, panPixZ }`.

**Global base scale** (`getGlobalBaseScale`): one pixels-per-physical-unit scale satisfying all 6 constraints (2 per panel × 3 panels):

```
baseScale = min(
    panelWidth  / (X * sx),    panelHeight / (Y * sy),
    panelHeight / (Z * sz),    panelWidth  / (Z * sz)
)
```

This ensures the X axis renders at the **same pixel width** in axial and coronal, Y at the same height in axial and sagittal, etc.

**Per-axis pan mapping**:
- Axial: horizontal=panPixX, vertical=panPixY
- Coronal: horizontal=panPixX, vertical=panPixZ
- Sagittal: horizontal=panPixZ, vertical=panPixY

Panning in one panel updates the shared axis offsets → all three panels redraw.

#### 4.4 Crosshairs

Thin coloured lines drawn on each canvas at the intersection coordinates. Clicking a panel sets the crosshair (slice indices) for the other two panels.

#### 4.5 Selection

Left-click a segment → `toggleSelection(id)` adds/removes from `NV.selected` Set → redraws all slices with highlighting → calls `_set3dSelection` to update 3D splats.

Selection chips rendered in toolbar as clickable pills with × close buttons.

### Step 5 — 3D Gaussian Splat Renderer (`volume_renderer.js`)

**Goal**: Render selected segmentation instances as translucent Gaussian splat point clouds, with orthogonal slice planes for context.

#### 5.1 Data loading

Fetches the same downsampled chunks as app.js (at ds ≈ maxDim/256):
- Raw uint8 chunk → 3D texture for slice planes
- Seg RGBA chunk → 3D texture for slice planes + splat colours
- Seg ID chunk → 3D texture for selection lookup + splat extraction (kept in original ZYX order)

`reorderZYX()` converts from server's ZYX C-order to Three.js texture layout (X-fastest).

#### 5.2 Physical coordinate system

```
pX = dx * sx,  pY = dy * sy,  pZ = dz * sz
M  = max(pX, pY, pZ)
nx = pX/M,  ny = pY/M,  nz = pZ/M    // scene coordinates [0, nx] × [0, ny] × [0, nz]
```

The bounding box spans (0,0,0) to (nx, ny, nz). Camera orbits around the centre.

#### 5.3 Gaussian splat shaders (GLSL 300 ES)

**Vertex shader**: positions each voxel as a screen-aligned point. Size is computed using the projection matrix for correct perspective scaling:

```glsl
gl_PointSize = clamp(uVoxelSize * projectionMatrix[1][1] * uScreenHeight / dist, 1.0, 256.0);
```

- `uVoxelSize` = geometric mean of voxel dimensions × 2.0 (for overlap)
- `projectionMatrix[1][1]` = `1/tan(fov/2)` — perspective-correct scaling
- Clamped to [1, 256] pixels to prevent explosions at close range

**Fragment shader**: applies Gaussian falloff per-point:

```glsl
float gauss = exp(-d2 * 14.0);      // steep falloff
float a = gauss * gauss * uOpacity;  // quadratic alpha for transparent interiors
```

The quadratic alpha (`gauss²`) is key — it makes the interior of dense segments translucent while surface splats (fewer overlaps) remain visible.

#### 5.4 Splat generation (`buildSplats`)

On selection change:
1. Count matching voxels in the seg ID volume
2. Subsample if > 300K (uniform stride)
3. Extract (x, y, z) positions + RGB colours into flat Float32Arrays
4. Create `THREE.BufferGeometry` + `THREE.Points`

Coordinates: `scene_x = (voxel_x + 0.5) / dx * nx` — normalised to scene bounds.

#### 5.5 Orthogonal slice planes

Three planes built with explicit `BufferGeometry` (not PlaneGeometry + rotation — avoids UV mapping bugs with anisotropic data):

- Axial (XY at Z): vertices span [-nx/2, nx/2] × [-ny/2, ny/2], UV maps to texture (u → X, v → Y)
- Coronal (XZ at Y): vertices span [-nx/2, nx/2] × [-nz/2, nz/2], UV maps to (u → X, v → Z)
- Sagittal (YZ at X): vertices span [-ny/2, ny/2] × [-nz/2, nz/2], UV maps to (u → Y, v → Z)

Fragment shader samples the 3D raw+seg textures at the correct slice fraction.

Planes are hidden when segments are selected (splats replace them).

#### 5.6 Interaction bridges

`window._updateSlicePlanes` — called by app.js when slice indices change.
`window._set3dSegOverlay(show, opacity)` — called on overlay toggle.
`window._set3dSelection(ids)` — called on selection change; rebuilds splats.

### Step 6 — CLI Entry Point (`__main__.py`)

```bash
python -m neurons.visualizer --raw data/raw.h5 --seg data/seg.h5 --spacing 30,6,6
```

- Loads volumes via `volume_loader.load_volume`
- Registers in `_store["default"]` with palette
- Starts uvicorn on port 8899
- Auto-opens browser (disable with `--no-browser`)

## Dependencies

```
fastapi
uvicorn[standard]
Pillow          # PNG encoding for slice endpoint
numpy
```

Optional: `scikit-image` (for marching-cubes mesh extraction, currently unused).
Three.js 0.158.0 loaded from CDN.

## Key Design Decisions

1. **Client-side slice extraction** — volumes are pre-downloaded as typed arrays; slicing + overlay happen in JS with zero latency. Server PNG endpoint exists but is unused in the current flow.

2. **Global base scale for co-axis alignment** — a single pixels-per-physical-unit scale ensures shared axes (e.g., X in axial + coronal) render at identical pixel widths.

3. **Sagittal transpose** — sagittal panel uses rows=Y, cols=Z (not the natural rows=Z, cols=Y) so that its vertical axis (Y) aligns with axial's vertical axis when panels are side by side.

4. **Gaussian splatting over mesh rendering** — each voxel rendered as a soft Gaussian blob gives a natural translucent volumetric look without requiring server-side marching cubes. Runs entirely on the client from the already-loaded seg volume.

5. **Selection-driven 3D** — only selected segments appear in 3D. Slice planes show when nothing is selected; splats replace them when segments are active. This keeps the 3D view uncluttered.

6. **GLSL 300 ES with RawShaderMaterial** — explicit shader control avoids Three.js auto-conversion pitfalls with WebGL2. All shaders use `in`/`out`, explicit `out vec4 fragColor`, and declare all uniforms/attributes.
