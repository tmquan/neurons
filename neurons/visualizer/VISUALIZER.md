# Neurons Volume Visualizer — Implementation Guide

A web-based neuroglancer-style volume viewer with 4-panel layout (axial, coronal, sagittal, 3D anisotropic Gaussian splats), built with FastAPI + vanilla JS + Three.js.

## Architecture

```
neurons/visualizer/
  __init__.py              # package marker
  __main__.py              # CLI: python -m neurons.visualizer --raw X --seg Y
  volume_loader.py         # unified loader for HDF5/TIFF/NRRD/NIfTI/NumPy
  app.py                   # FastAPI REST endpoints + PNG composition
  static/
    index.html             # single-page shell, 2×2 grid + toolbar
    style.css              # light-gray theme, scientific visualization style
    app.js                 # 2D slice rendering, crosshairs, selection, boot
    volume_renderer.js     # Three.js 3D panel: anisotropic Gaussian splats + slice planes
```

```
┌─────────────────────────────────────────────────┐
│                  Browser (SPA)                   │
│ ┌───────────┬───────────┐                        │
│ │ Axial XY  │ Sagittal  │  app.js: slice extract │
│ │ (canvas)  │ ZY(canvas)│  + overlay composite   │
│ ├───────────┼───────────┤                        │
│ │ Coronal XZ│ 3D Splats │  volume_renderer.js:   │
│ │ (canvas)  │ (WebGL2)  │  Three.js + GLSL       │
│ └───────────┴───────────┘                        │
│ [Overlay][Opacity][Filled][Colormap][Instance]   │
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
| `GET  .../seg_id_chunk?downsample=N` | uint8 bytes | Seg IDs mod 256 for selection |
| `GET  .../labels` | JSON label list + palette | Unique IDs + RGB colours |

**Key decisions**:
- The `chunk` and `seg_chunk` endpoints downsample server-side so the browser receives manageable arrays (~256³).
- Seg IDs are reduced to uint8 (mod 256) for GPU texture lookup — sufficient for up to 255 distinct segments.
- A random RGB palette is generated per-session with seed=0 for reproducibility.
- The slice endpoint composites raw grayscale + seg overlay using alpha blending, returning PNG.

**In-memory store**: `_store[vid] = _VolumeEntry(raw, seg, palette)`. The CLI (`__main__.py`) pre-loads volumes with `vid="default"`.

### Step 3 — HTML Shell + CSS (`index.html`, `style.css`)

**Goal**: 2×2 panel grid with a toolbar, light-gray scientific visualization theme.

```
┌──────────────┬──────────────┐
│   Axial (XY) │ Sagittal(YZ) │
├──────────────┼──────────────┤
│ Coronal (XZ) │  3D Volume   │
└──────────────┴──────────────┘
[ Overlay ✓ ] [ Opacity ━━━ ] [ Filled ✓ ] [ Colormap ▾ ] [ Instance: #N ]
```

- CSS Grid 2×2 layout, each cell contains a `<canvas>`.
- Light-gray theme (`#f0f0f4` panels, `#e0e0e6` toolbar) matching publication-quality scientific visualizations.
- Toolbar with overlay toggle, opacity slider, filled/surface toggle, colormap select, instance info, coordinate display.
- Selection chips: blue pills with × to deselect, "clear" link.
- Three.js 0.158.0 loaded from CDN.

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

Blue (vertical) and orange (horizontal) lines drawn on each canvas at the intersection coordinates. Clicking a panel sets the crosshair (slice indices) for the other two panels.

#### 4.5 Selection

Left-click a segment → `toggleSelection(id)` adds/removes from `NV.selected` Set → redraws all slices with highlighting → calls `_set3dSelection` to update 3D splats.

Selection chips rendered in toolbar as clickable pills with × close buttons.

### Step 5 — 3D Anisotropic Gaussian Splat Renderer (`volume_renderer.js`)

**Goal**: Render selected segmentation instances as translucent anisotropic Gaussian splat point clouds, with orthogonal slice planes for context when no segments are selected.

#### 5.1 Data loading

Fetches the same downsampled chunks as app.js (at ds ≈ maxDim/256):
- Raw uint8 chunk → 3D texture for slice planes
- Seg RGBA chunk → 3D texture for slice planes + splat colours (kept in original ZYX order for voxel extraction)
- Seg ID chunk → 3D texture for selection lookup + splat extraction (kept in original ZYX order)

`reorderZYX()` converts from server's ZYX C-order to Three.js texture layout (X-fastest) for the 3D textures.

#### 5.2 Physical coordinate system

```
pX = dx * sx,  pY = dy * sy,  pZ = dz * sz
M  = max(pX, pY, pZ)
nx = pX/M,  ny = pY/M,  nz = pZ/M    // scene coordinates [0, nx] × [0, ny] × [0, nz]
```

The bounding box spans (0,0,0) to (nx, ny, nz). Camera orbits around the centre.

#### 5.3 Anisotropic Gaussian splat shaders (GLSL 300 ES)

Each voxel is rendered as a **3D ellipsoidal Gaussian** projected to a 2D screen-space ellipse. This is essential for anisotropic data (e.g., Z spacing 10× larger than XY).

**Per-axis sigma values**: `σx = (nx/dx) × 0.75`, `σy = (ny/dy) × 0.75`, `σz = (nz/dz) × 0.75` — each proportional to the voxel's physical size along that axis.

**Vertex shader** — projects 3D covariance to 2D screen ellipse:

```glsl
// Transform 3 scaled basis vectors to view space
mat3 R = mat3(modelViewMatrix);
vec3 ax = σx * R[0],  ay = σy * R[1],  az = σz * R[2];

// 2D screen-space covariance (pixels²)
cxx = (ax.x² + ay.x² + az.x²) × (focal/z)²
cxy = (ax.x·ax.y + ay.x·ay.y + az.x·az.y) × (focal/z)²
cyy = (ax.y² + ay.y² + az.y²) × (focal/z)²

// Point size from major eigenvalue (covers 3σ)
λ_max = mid + √(mid² - det)
gl_PointSize = clamp(6√λ_max, 1, 512)

// Pass inverse covariance to fragment shader
vInvCov = (1/det) × (cyy, -cxy, cxx)
```

**Fragment shader** — evaluates anisotropic Gaussian per-pixel:

```glsl
vec2 d = (gl_PointCoord - 0.5) × vPtSize;     // pixel offset from centre
float maha = d^T × Σ⁻¹ × d;                   // Mahalanobis distance
float gauss = exp(-0.5 × maha);                // elliptical Gaussian
fragColor = vec4(color × shading, gauss × opacity);
```

**Critical**: `gl_PointSize` is vertex-shader-only in GLSL 300 ES. The point size must be passed to the fragment shader via `flat out float vPtSize`.

The result: when viewed from the side, splats stretch along Z (matching anisotropy); from the top, they're round (isotropic XY).

#### 5.4 Splat generation (`buildSplats`)

On selection change:
1. Iterate the seg ID volume (ZYX order, on client)
2. **Filled mode**: include all voxels of selected segments
3. **Surface mode** (Filled unchecked): include only voxels where at least one 6-connected neighbor has a different ID — gives a hollow shell appearance
4. Subsample if > 300K voxels (uniform stride)
5. Extract (x, y, z) positions + RGB colours into `Float32Array`s
6. Create `THREE.BufferGeometry` + `THREE.Points` with `RawShaderMaterial`

Coordinates: `scene_x = (voxel_x + 0.5) / dx × nx` — normalised to scene bounds.

Surface detection (`isSurface`): checks 6 neighbors (±X, ±Y, ±Z). Boundary voxels and volume-edge voxels are always considered surface.

#### 5.5 Orthogonal slice planes

Three planes built with explicit `BufferGeometry` (not PlaneGeometry + rotation — avoids UV mapping bugs with anisotropic data):

- Axial (XY at Z): UV maps to texture (u → X, v → Y)
- Coronal (XZ at Y): UV maps to (u → X, v → Z)
- Sagittal (YZ at X): UV maps to (u → Y, v → Z)

Fragment shader samples the 3D raw+seg textures at the correct slice fraction, with selection-based opacity modulation.

Planes are **hidden when segments are selected** (splats replace them); visible otherwise.

#### 5.6 Lighting

Studio-style lighting for the light-gray theme:
- **Hemisphere light** (white sky / soft gray ground) — natural ambient
- **Key directional** (top-right, intensity 1.2) — primary illumination
- **Fill directional** (left, intensity 0.4) — soften shadows
- **Rim directional** (below, intensity 0.3) — edge definition

Splat shading in the fragment shader adds a subtle specular highlight at the Gaussian centre for a polished, publication-quality look.

#### 5.7 Interaction bridges

| Function | Called by | Purpose |
|---|---|---|
| `window._updateSlicePlanes()` | app.js on scroll | Sync 3D plane positions |
| `window._set3dSegOverlay(show, opacity)` | Overlay checkbox/slider | Toggle seg on planes |
| `window._set3dSelection(ids)` | Selection change | Rebuild Gaussian splats |
| `window._set3dFilled(bool)` | Filled checkbox | Toggle filled vs surface-only |

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

Three.js 0.158.0 loaded from CDN. No other client-side dependencies.

## Key Design Decisions

1. **Client-side slice extraction** — volumes are pre-downloaded as typed arrays; slicing + overlay happen in JS with zero latency. Server PNG endpoint exists but is unused in the current flow.

2. **Global base scale for co-axis alignment** — a single pixels-per-physical-unit scale ensures shared axes (e.g., X in axial + coronal) render at identical pixel widths. Zoom and pan are synchronized across all three panels.

3. **Sagittal transpose** — sagittal panel uses rows=Y, cols=Z (not the natural rows=Z, cols=Y) so that its vertical axis (Y) aligns with axial's vertical axis when panels are side by side.

4. **Anisotropic 3D Gaussian splatting** — each voxel is a 3D ellipsoidal Gaussian with per-axis sigma matching the voxel spacing. The 3D covariance is projected to a 2D screen-space ellipse via the view matrix and perspective Jacobian — the standard 3DGS projection. This fills Z-gaps in anisotropic EM data (e.g., 40nm Z vs 4nm XY) without over-blurring in XY.

5. **Filled vs surface rendering** — "Filled" mode includes all segment voxels as splats (translucent volumetric cloud). "Surface" mode (Filled unchecked) keeps only boundary voxels (6-connected neighbor test), giving a hollow shell with fewer points and clearer shape.

6. **Selection-driven 3D** — only selected segments appear in 3D. Slice planes show when nothing is selected; splats replace them when segments are active. This keeps the 3D view uncluttered and focused.

7. **GLSL 300 ES with RawShaderMaterial** — explicit shader control avoids Three.js auto-conversion pitfalls with WebGL2. All shaders use `in`/`out`, explicit `out vec4 fragColor`, and declare all uniforms/attributes. `gl_PointSize` is passed to the fragment shader via `flat out float` since it's vertex-shader-only in GLSL 300 ES.

8. **Light-gray scientific theme** — light backgrounds (`#f0f0f4`) with studio-style 3-point lighting match publication-quality scientific visualizations (similar to CellPACK / Allen Institute renderings). Blue/orange crosshairs remain visible on light panels.
