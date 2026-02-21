/**
 * Three.js 3-D panel: Gaussian-splatted segmentation rendering
 * + 3 orthogonal slice planes.
 *
 * Selected segments are rendered as point-cloud Gaussian splats —
 * each voxel becomes a soft, screen-aligned Gaussian blob, giving
 * a translucent volumetric appearance.
 */

"use strict";

/* ── Anisotropic 3D Gaussian splat shaders (GLSL 300 ES) ────── */

const SPLAT_VERT = `
in vec3 position;
in vec3 color;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 uSigma;
uniform float uScreenHeight;
out vec3 vColor;
flat out vec3 vInvCov;
flat out float vPtSize;

void main() {
    vColor = color;
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    float z = max(-mvPos.z, 0.01);

    mat3 R = mat3(modelViewMatrix);
    vec3 ax = uSigma.x * R[0];
    vec3 ay = uSigma.y * R[1];
    vec3 az = uSigma.z * R[2];

    float cxx = ax.x*ax.x + ay.x*ay.x + az.x*az.x;
    float cxy = ax.x*ax.y + ay.x*ay.y + az.x*az.y;
    float cyy = ax.y*ax.y + ay.y*ay.y + az.y*az.y;

    float focal = projectionMatrix[1][1] * uScreenHeight * 0.5;
    float s = focal / z;
    cxx *= s * s;  cxy *= s * s;  cyy *= s * s;

    float mid = 0.5 * (cxx + cyy);
    float det = cxx * cyy - cxy * cxy;
    float disc = max(mid * mid - det, 0.0);
    float lambda_max = mid + sqrt(disc);
    float radius = ceil(3.0 * sqrt(max(lambda_max, 0.1)));
    float ptSz = clamp(2.0 * radius, 1.0, 512.0);
    gl_PointSize = ptSz;
    vPtSize = ptSz;

    float invDet = 1.0 / max(det, 1e-6);
    vInvCov = vec3(cyy * invDet, -cxy * invDet, cxx * invDet);

    gl_Position = projectionMatrix * mvPos;
}
`;

const SPLAT_FRAG = `
precision highp float;
uniform float uOpacity;
in vec3 vColor;
flat in vec3 vInvCov;
flat in float vPtSize;
out vec4 fragColor;
void main() {
    vec2 d = vec2(gl_PointCoord.x - 0.5, 0.5 - gl_PointCoord.y) * vPtSize;
    float maha = d.x * d.x * vInvCov.x + 2.0 * d.x * d.y * vInvCov.y + d.y * d.y * vInvCov.z;
    float gauss = exp(-0.5 * maha);
    if (gauss < 0.02) discard;
    float a = gauss * uOpacity;
    vec3 lit = vColor * (0.55 + 0.45 * gauss) + vec3(0.12 * gauss * gauss);
    fragColor = vec4(lit, a);
}
`;

/* ── GLSL 300 ES — slice plane ────────────────────────────── */

const PLANE_VERT = `
in vec3 position;
in vec2 uv;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
out vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const PLANE_FRAG = `
precision highp float;
precision highp sampler3D;
precision highp sampler2D;
in vec2 vUv;
out vec4 fragColor;

uniform sampler3D uVolume;
uniform sampler3D uSeg;
uniform sampler3D uSegId;
uniform sampler2D uSelMask;
uniform float uSelMaskSize;
uniform bool  uHasSelection;
uniform int   uAxis;
uniform float uSliceFrac;
uniform float uWinLo;
uniform float uWinHi;
uniform float uSegOpacity;
uniform bool  uShowSeg;

void main() {
    vec3 tc;
    if (uAxis == 0)      tc = vec3(vUv.x, vUv.y, uSliceFrac);
    else if (uAxis == 1) tc = vec3(vUv.x, uSliceFrac, vUv.y);
    else                 tc = vec3(uSliceFrac, vUv.x, vUv.y);

    float raw = texture(uVolume, tc).r;
    float v = clamp((raw - uWinLo) / max(uWinHi - uWinLo, 0.001), 0.0, 1.0);
    vec3 c = vec3(v);

    if (uShowSeg) {
        vec4 seg = texture(uSeg, tc);
        if (seg.a > 0.01) {
            float segIdVal = texture(uSegId, tc).r * 255.0;

            bool isSel = false;
            if (uHasSelection && uSelMaskSize > 0.0) {
                float u = (segIdVal + 0.5) / uSelMaskSize;
                isSel = texture(uSelMask, vec2(u, 0.5)).r > 0.5;
            }

            float strength = uHasSelection
                ? (isSel ? uSegOpacity : uSegOpacity * 0.12)
                : uSegOpacity;
            c = mix(c, seg.rgb, strength);
        }
    }

    fragColor = vec4(c, 0.92);
}
`;

/* ── helpers ───────────────────────────────────────────────── */

function reorderZYX(data, dz, dy, dx) {
    const out = new Uint8Array(dx * dy * dz);
    for (let z = 0; z < dz; z++)
        for (let y = 0; y < dy; y++)
            for (let x = 0; x < dx; x++)
                out[x + y * dx + z * dx * dy] = data[z * dy * dx + y * dx + x];
    return out;
}

function reorderZYX_RGBA(data, dz, dy, dx) {
    const out = new Uint8Array(dx * dy * dz * 4);
    for (let z = 0; z < dz; z++)
        for (let y = 0; y < dy; y++)
            for (let x = 0; x < dx; x++) {
                const srcOff = (z * dy * dx + y * dx + x) * 4;
                const dstOff = (x + y * dx + z * dx * dy) * 4;
                out[dstOff]     = data[srcOff];
                out[dstOff + 1] = data[srcOff + 1];
                out[dstOff + 2] = data[srcOff + 2];
                out[dstOff + 3] = data[srcOff + 3];
            }
    return out;
}

/* ── init ──────────────────────────────────────────────────── */

async function initVolumeRenderer() {
    const st = window.NV;
    if (!st.ready) return;

    const container = document.getElementById("panel-3d");
    const canvas3d  = document.getElementById("canvas-3d");

    const testCtx = canvas3d.getContext("webgl2");
    if (!testCtx) {
        container.querySelector(".panel-label").textContent = "3D (WebGL2 required)";
        return;
    }

    const maxDim = Math.max(...st.shape);
    const ds = Math.max(1, Math.round(maxDim / 256));

    const cf = typeof cachedFetch === "function" ? cachedFetch : async (u) => fetch(u);

    /* ── fetch raw chunk ──────────────────────────────────────── */
    let volData, volShape, volSpacing;
    try {
        const r = await cf(`/api/volumes/${st.vid}/chunk?downsample=${ds}`, "3D raw chunk");
        volShape   = r.headers.get("X-Shape").split(",").map(Number);
        volSpacing = r.headers.get("X-Spacing").split(",").map(Number);
        volData = new Uint8Array(await r.arrayBuffer());
    } catch (e) {
        console.error("3D chunk load failed", e);
        container.querySelector(".panel-label").textContent = "3D: load failed";
        return;
    }

    const [dz, dy, dx] = volShape;
    const [sz, sy, sx] = volSpacing;

    /* ── fetch seg chunks ─────────────────────────────────────── */
    let segData = null, segIdData = null;
    let hasSeg = false;
    try {
        const r2 = await cf(`/api/volumes/${st.vid}/seg_chunk?downsample=${ds}`, "3D seg chunk");
        if (r2.ok) {
            segData = new Uint8Array(await r2.arrayBuffer());
            hasSeg = true;
        }
    } catch (_) {}

    try {
        const r3 = await cf(`/api/volumes/${st.vid}/seg_id_chunk?downsample=${ds}`, "3D seg IDs");
        if (r3.ok) {
            segIdData = new Uint8Array(await r3.arrayBuffer());
        }
    } catch (_) {}

    /* ── reorder for Three.js textures ────────────────────────── */
    const rawTex = reorderZYX(volData, dz, dy, dx);
    const segTex = hasSeg ? reorderZYX_RGBA(segData, dz, dy, dx) : new Uint8Array(dx * dy * dz * 4);
    const segIdTex = segIdData ? reorderZYX(segIdData, dz, dy, dx) : new Uint8Array(dx * dy * dz);

    /* ── physical extents ─────────────────────────────────────── */
    const pX = dx * sx, pY = dy * sy, pZ = dz * sz;
    const M = Math.max(pX, pY, pZ);
    const nx = pX / M, ny = pY / M, nz = pZ / M;

    /* ── Three.js scene ───────────────────────────────────────── */
    const rect = container.getBoundingClientRect();
    const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(rect.width, rect.height);

    const scene  = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f4);
    const camera = new THREE.PerspectiveCamera(50, rect.width / rect.height, 0.001, 50);

    /* ── 3D textures (for slice planes) ───────────────────────── */
    const Tex3D = THREE.Data3DTexture || THREE.DataTexture3D;

    const tRaw = new Tex3D(rawTex, dx, dy, dz);
    tRaw.format = THREE.RedFormat; tRaw.type = THREE.UnsignedByteType;
    tRaw.minFilter = THREE.LinearFilter; tRaw.magFilter = THREE.LinearFilter;
    tRaw.unpackAlignment = 1; tRaw.needsUpdate = true;

    const tSeg = new Tex3D(segTex, dx, dy, dz);
    tSeg.format = THREE.RGBAFormat; tSeg.type = THREE.UnsignedByteType;
    tSeg.minFilter = THREE.NearestFilter; tSeg.magFilter = THREE.NearestFilter;
    tSeg.unpackAlignment = 1; tSeg.needsUpdate = true;

    const tSegId = new Tex3D(segIdTex, dx, dy, dz);
    tSegId.format = THREE.RedFormat; tSegId.type = THREE.UnsignedByteType;
    tSegId.minFilter = THREE.NearestFilter; tSegId.magFilter = THREE.NearestFilter;
    tSegId.unpackAlignment = 1; tSegId.needsUpdate = true;

    const SEL_SIZE = 256;
    const selMaskData = new Uint8Array(SEL_SIZE);
    const tSelMask = new THREE.DataTexture(selMaskData, SEL_SIZE, 1, THREE.RedFormat, THREE.UnsignedByteType);
    tSelMask.minFilter = THREE.NearestFilter; tSelMask.magFilter = THREE.NearestFilter;
    tSelMask.needsUpdate = true;

    /* ── auto-window ──────────────────────────────────────────── */
    const sorted = Array.from(volData).sort((a, b) => a - b);
    const wLo = sorted[Math.floor(sorted.length * 0.02)] / 255;
    const wHi = sorted[Math.floor(sorted.length * 0.98)] / 255;

    const selUniforms = {
        uSegId:       { value: tSegId },
        uSelMask:     { value: tSelMask },
        uSelMaskSize: { value: SEL_SIZE },
        uHasSelection:{ value: false },
    };

    /* ── lighting (soft studio style) ────────────────────────── */
    scene.add(new THREE.HemisphereLight(0xffffff, 0xd0d0d8, 0.7));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(2, 3, 2);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xe8e8f0, 0.4);
    fillLight.position.set(-2, 0.5, -1);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
    rimLight.position.set(0, -2, 1);
    scene.add(rimLight);

    /* ── Gaussian splat state ─────────────────────────────────── */
    let splatPoints = null;
    const sigmaX = (nx / dx) * 0.75;
    const sigmaY = (ny / dy) * 0.75;
    const sigmaZ = (nz / dz) * 0.75;

    const splatMat = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: SPLAT_VERT,
        fragmentShader: SPLAT_FRAG,
        uniforms: {
            uSigma:        { value: new THREE.Vector3(sigmaX, sigmaY, sigmaZ) },
            uScreenHeight: { value: rect.height },
            uOpacity:      { value: 0.15 },
        },
        transparent: true,
        depthWrite: false,
        depthTest: true,
        blending: THREE.NormalBlending,
    });

    function isSurface(off, id, x, y, z) {
        if (x === 0 || x === dx - 1 || y === 0 || y === dy - 1 || z === 0 || z === dz - 1) return true;
        const s = dy * dx;
        return segIdData[off - 1]  !== id || segIdData[off + 1]  !== id ||
               segIdData[off - dx] !== id || segIdData[off + dx] !== id ||
               segIdData[off - s]  !== id || segIdData[off + s]  !== id;
    }

    let splatFilled = true;

    function buildSplats(selectedIdSet) {
        if (splatPoints) {
            scene.remove(splatPoints);
            splatPoints.geometry.dispose();
            splatPoints = null;
        }
        if (!segIdData || !segData || selectedIdSet.size === 0) return;

        const filled = splatFilled;
        let totalVox = 0;
        for (let z = 0; z < dz; z++)
            for (let y = 0; y < dy; y++)
                for (let x = 0; x < dx; x++) {
                    const off = z * dy * dx + y * dx + x;
                    const id = segIdData[off];
                    if (!selectedIdSet.has(id)) continue;
                    if (!filled && !isSurface(off, id, x, y, z)) continue;
                    totalVox++;
                }
        if (totalVox === 0) return;

        const MAX_SPLATS = 300000;
        const stride = Math.max(1, Math.ceil(totalVox / MAX_SPLATS));
        const estCount = Math.ceil(totalVox / stride);

        const positions = new Float32Array(estCount * 3);
        const colors    = new Float32Array(estCount * 3);
        let cnt = 0, skip = 0;

        for (let z = 0; z < dz; z++) {
            for (let y = 0; y < dy; y++) {
                for (let x = 0; x < dx; x++) {
                    const off = z * dy * dx + y * dx + x;
                    const id = segIdData[off];
                    if (!selectedIdSet.has(id)) continue;
                    if (!filled && !isSurface(off, id, x, y, z)) continue;

                    skip++;
                    if (stride > 1 && skip % stride !== 0) continue;

                    positions[cnt * 3]     = (x + 0.5) / dx * nx;
                    positions[cnt * 3 + 1] = (y + 0.5) / dy * ny;
                    positions[cnt * 3 + 2] = (z + 0.5) / dz * nz;

                    const rgbaOff = off * 4;
                    colors[cnt * 3]     = segData[rgbaOff]     / 255;
                    colors[cnt * 3 + 1] = segData[rgbaOff + 1] / 255;
                    colors[cnt * 3 + 2] = segData[rgbaOff + 2] / 255;

                    cnt++;
                    if (cnt >= estCount) break;
                }
                if (cnt >= estCount) break;
            }
            if (cnt >= estCount) break;
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(positions.slice(0, cnt * 3), 3));
        geom.setAttribute("color",    new THREE.BufferAttribute(colors.slice(0, cnt * 3), 3));

        splatMat.uniforms.uOpacity.value = filled ? 0.15 : 0.35;
        splatPoints = new THREE.Points(geom, splatMat);
        scene.add(splatPoints);
        console.log(`Gaussian splats: ${cnt} points (${filled ? "filled" : "surface"}, stride ${stride})`);
    }

    /* ── 3 orthogonal slice planes ────────────────────────────── */
    const sharedPlane = {
        uVolume:     { value: tRaw },
        uSeg:        { value: tSeg },
        ...selUniforms,
        uWinLo:      { value: wLo },
        uWinHi:      { value: wHi },
        uSegOpacity: { value: 0.7 },
        uShowSeg:    { value: hasSeg },
    };

    function makePlane(axis) {
        const geom = new THREE.BufferGeometry();
        let positions;
        if (axis === 0) {
            positions = new Float32Array([
                -nx/2, -ny/2, 0,   nx/2, -ny/2, 0,
                -nx/2,  ny/2, 0,   nx/2,  ny/2, 0,
            ]);
        } else if (axis === 1) {
            positions = new Float32Array([
                -nx/2, 0, -nz/2,   nx/2, 0, -nz/2,
                -nx/2, 0,  nz/2,   nx/2, 0,  nz/2,
            ]);
        } else {
            positions = new Float32Array([
                0, -ny/2, -nz/2,   0, ny/2, -nz/2,
                0, -ny/2,  nz/2,   0, ny/2,  nz/2,
            ]);
        }
        const uvs = new Float32Array([0,0, 1,0, 0,1, 1,1]);
        geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geom.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
        geom.setIndex([0, 1, 2, 2, 1, 3]);

        const mat = new THREE.RawShaderMaterial({
            glslVersion: THREE.GLSL3,
            vertexShader: PLANE_VERT, fragmentShader: PLANE_FRAG,
            uniforms: {
                ...sharedPlane,
                uAxis:      { value: axis },
                uSliceFrac: { value: 0.5 },
            },
            side: THREE.DoubleSide, transparent: true,
        });

        const mesh = new THREE.Mesh(geom, mat);
        mesh.renderOrder = 10;
        scene.add(mesh);
        return { mesh, mat };
    }

    const planes = [makePlane(0), makePlane(1), makePlane(2)];

    function updatePlanes() {
        const s = st.sliceIdx;
        const fZ = (s[0] + 0.5) / st.shape[0];
        const fY = (s[1] + 0.5) / st.shape[1];
        const fX = (s[2] + 0.5) / st.shape[2];

        planes[0].mat.uniforms.uSliceFrac.value = fZ;
        planes[0].mesh.position.set(nx / 2, ny / 2, fZ * nz);
        planes[1].mat.uniforms.uSliceFrac.value = fY;
        planes[1].mesh.position.set(nx / 2, fY * ny, nz / 2);
        planes[2].mat.uniforms.uSliceFrac.value = fX;
        planes[2].mesh.position.set(fX * nx, ny / 2, nz / 2);
    }
    updatePlanes();

    window._updateSlicePlanes = updatePlanes;

    window._set3dSegOverlay = function(show, opacity) {
        planes.forEach(p => {
            p.mat.uniforms.uShowSeg.value = show;
            p.mat.uniforms.uSegOpacity.value = opacity;
        });
    };

    window._set3dSelection = function(selectedIds) {
        selMaskData.fill(0);
        for (const id of selectedIds) {
            const idx = id % SEL_SIZE;
            if (idx >= 0 && idx < SEL_SIZE) selMaskData[idx] = 255;
        }
        tSelMask.needsUpdate = true;

        const hasSel = selectedIds.length > 0;
        planes.forEach(p => {
            p.mat.uniforms.uHasSelection.value = hasSel;
            p.mesh.visible = !hasSel;
        });

        buildSplats(new Set(selectedIds));
    };

    window._set3dFilled = function(filled) {
        splatFilled = filled;
        console.log("Filled mode:", filled);
        const st = window.NV;
        if (st.selected.size > 0) {
            buildSplats(new Set(st.selected));
        }
    };

    /* ── bounding box wireframe ───────────────────────────────── */
    const bbGeom = new THREE.BoxGeometry(nx, ny, nz);
    const bbEdge = new THREE.EdgesGeometry(bbGeom);
    const bbLine = new THREE.LineSegments(bbEdge, new THREE.LineBasicMaterial({ color: 0xb0b0b8 }));
    bbLine.position.set(nx / 2, ny / 2, nz / 2);
    scene.add(bbLine);

    /* ── axis indicator ───────────────────────────────────────── */
    const axLen = 0.12;
    const axHelper = new THREE.Group();
    axHelper.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(axLen,0,0)]),
        new THREE.LineBasicMaterial({ color: 0xcc3333 })
    ));
    axHelper.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,axLen,0)]),
        new THREE.LineBasicMaterial({ color: 0x33aa33 })
    ));
    axHelper.add(new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,axLen)]),
        new THREE.LineBasicMaterial({ color: 0x3366cc })
    ));
    axHelper.position.set(-0.05, -0.05, -0.05);
    scene.add(axHelper);

    console.log("Splat renderer initialised. Box:", nx.toFixed(3), ny.toFixed(3), nz.toFixed(3));
    container.querySelector(".panel-label").textContent = "3D Volume";

    /* ── orbit / pan / zoom ───────────────────────────────────── */
    let dragMode = null, prevX = 0, prevY = 0;
    let theta = 0.8, phi = 1.0, radius = 1.8;
    const target = new THREE.Vector3(nx / 2, ny / 2, nz / 2);

    function updateCamera() {
        camera.position.set(
            target.x + radius * Math.sin(phi) * Math.cos(theta),
            target.y + radius * Math.cos(phi),
            target.z + radius * Math.sin(phi) * Math.sin(theta),
        );
        camera.lookAt(target);
    }
    updateCamera();

    canvas3d.addEventListener("contextmenu", e => e.preventDefault());
    canvas3d.addEventListener("mousedown", e => {
        prevX = e.clientX; prevY = e.clientY;
        if (e.button === 0)      dragMode = "rotate";
        else if (e.button === 2) dragMode = "pan";
    });
    window.addEventListener("mouseup", () => { dragMode = null; });
    canvas3d.addEventListener("mousemove", e => {
        if (!dragMode) return;
        const ddx = e.clientX - prevX, ddy = e.clientY - prevY;
        prevX = e.clientX; prevY = e.clientY;
        if (dragMode === "rotate") {
            theta += ddx * 0.008;
            phi   -= ddy * 0.008;
            phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi));
        } else {
            const sp = radius * 0.002;
            const right = new THREE.Vector3(), fwd = new THREE.Vector3();
            fwd.subVectors(target, camera.position).normalize();
            right.crossVectors(camera.up, fwd).normalize();
            const up = new THREE.Vector3().crossVectors(fwd, right).normalize();
            target.addScaledVector(right, -ddx * sp);
            target.addScaledVector(up, ddy * sp);
        }
        updateCamera();
    });
    canvas3d.addEventListener("wheel", e => {
        e.preventDefault();
        radius *= e.deltaY > 0 ? 1.1 : 0.91;
        radius = Math.max(0.1, Math.min(10, radius));
        updateCamera();
    }, { passive: false });

    /* ── render loop ──────────────────────────────────────────── */
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();

    new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        if (width > 0 && height > 0) {
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            splatMat.uniforms.uScreenHeight.value = height;
        }
    }).observe(container);
}
