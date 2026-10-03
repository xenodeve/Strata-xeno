import {
  BufferAttribute,
  BufferGeometry,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineLoop,
  PerspectiveCamera,
  Points,
  Scene,
  ShaderMaterial,
  WebGLRenderer,
} from "three";
import { EXPERTS, STRATA, allocate, mulberry32, stratumOf, type StratumId } from "./strata";

/** Layout of the core sample, in scene units. */
const SLAB_H = 1;
const GAP = 0.14;
const RADIUS = 1.5;
const TOTAL = STRATA.length * SLAB_H + (STRATA.length - 1) * GAP;

const COLOR: Record<StratumId, [number, number, number]> = {
  gpu1: [0.353, 0.635, 1.0], // #5aa2ff, the accent of the app's dark theme
  gpu2: [0.62, 0.79, 1.0],
  ram: [0.91, 0.91, 0.93],
  nvme: [0.55, 0.55, 0.59],
};
const SIZE: Record<StratumId, number> = { gpu1: 0.062, gpu2: 0.058, ram: 0.05, nvme: 0.046 };

const VERT = /* glsl */ `
  attribute vec3 aColor;
  attribute float aSize;
  uniform float uScale;
  varying vec3 vColor;
  varying float vFade;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = aSize * uScale / -mv.z;
    vColor = aColor;
    // dots at the back of the sample are dimmer, which is what makes it read as a volume
    vFade = clamp(1.0 - (-mv.z - 10.2) / 5.0, 0.28, 1.0);
  }
`;
const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vFade;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.36, d) * vFade;
    if (a < 0.02) discard;
    gl_FragColor = vec4(vColor, a);
  }
`;

export type SceneApi = {
  setState(vram: number, capacity: boolean): void;
  rotateBy(rad: number): void;
  setDragging(on: boolean): void;
  /** Throw the sample with this angular speed (rad/s); it slows down by itself. */
  fling(velocity: number): void;
  setActive(active: boolean): void;
  resize(): void;
  dispose(): void;
};

type Options = { reducedMotion: boolean; onLost: () => void };

/** Returns null when WebGL is not available. */
export function createScene(canvas: HTMLCanvasElement, opts: Options): SceneApi | null {
  let renderer: WebGLRenderer;
  try {
    renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch {
    return null;
  }
  renderer.setClearColor(0x000000, 0);

  const scene = new Scene();
  const group = new Group();
  scene.add(group);
  // the sample sits right of centre: the labels live in the space on its left
  group.position.x = 1.15;
  const camera = new PerspectiveCamera(26, 1, 0.1, 60);
  camera.position.set(0, 1.5, 12.4);
  camera.lookAt(0, 0, 0);

  // fixed random attributes of each expert: where in the disc, and how high in its slab
  const rnd = mulberry32(20261003);
  const theta = new Float32Array(EXPERTS);
  const radius = new Float32Array(EXPERTS);
  const lift = new Float32Array(EXPERTS);
  for (let i = 0; i < EXPERTS; i++) {
    theta[i] = rnd() * Math.PI * 2;
    radius[i] = RADIUS * Math.sqrt(rnd());
    lift[i] = rnd();
  }

  const slabBottom = (s: number) => TOTAL / 2 - s * (SLAB_H + GAP) - SLAB_H;
  const cur = new Float32Array(EXPERTS * 3);
  const tgt = new Float32Array(EXPERTS * 3);
  const colors = new Float32Array(EXPERTS * 3);
  const sizes = new Float32Array(EXPERTS);

  const geometry = new BufferGeometry();
  const posAttr = new BufferAttribute(cur, 3);
  geometry.setAttribute("position", posAttr);
  geometry.setAttribute("aColor", new BufferAttribute(colors, 3));
  geometry.setAttribute("aSize", new BufferAttribute(sizes, 1));

  const material = new ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthWrite: false,
    uniforms: { uScale: { value: 600 } },
  });
  const points = new Points(geometry, material);
  points.frustumCulled = false;
  group.add(points);

  // faint rings at the top and bottom of each stratum
  const ringMat = new LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.16 });
  const ringGeos: BufferGeometry[] = [];
  const ringPts: number[] = [];
  for (let k = 0; k < 96; k++) {
    const a = (k / 96) * Math.PI * 2;
    ringPts.push(Math.cos(a) * (RADIUS + 0.06), 0, Math.sin(a) * (RADIUS + 0.06));
  }
  for (let s = 0; s < STRATA.length; s++) {
    for (const y of [slabBottom(s), slabBottom(s) + SLAB_H]) {
      const g = new BufferGeometry();
      g.setAttribute("position", new Float32BufferAttribute(ringPts, 3));
      const ring = new LineLoop(g, ringMat);
      ring.position.y = y;
      group.add(ring);
      ringGeos.push(g);
    }
  }

  let vram = 0.55;
  let capacity = false;
  let snapNext = true;

  function retarget() {
    const a = allocate(vram, capacity);
    for (let i = 0; i < EXPERTS; i++) {
      const id = stratumOf(i, a);
      const s = STRATA.indexOf(id);
      tgt[i * 3] = Math.cos(theta[i]) * radius[i];
      tgt[i * 3 + 1] = slabBottom(s) + 0.05 + lift[i] * (SLAB_H - 0.1);
      tgt[i * 3 + 2] = Math.sin(theta[i]) * radius[i];
      const c = COLOR[id];
      colors[i * 3] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
      sizes[i] = SIZE[id];
    }
    (geometry.getAttribute("aColor") as BufferAttribute).needsUpdate = true;
    (geometry.getAttribute("aSize") as BufferAttribute).needsUpdate = true;
    if (snapNext || opts.reducedMotion) {
      cur.set(tgt);
      posAttr.needsUpdate = true;
      snapNext = false;
    }
  }
  retarget();

  // ── loop control: it runs only while something moves and the sample is on screen ──
  let raf = 0;
  let last = 0;
  let active = true;
  let dragging = false;
  let velocity = 0;
  const auto = !opts.reducedMotion;

  function stepPositions(dt: number): boolean {
    const k = 1 - Math.exp(-dt * 5.5);
    let moving = false;
    for (let i = 0; i < EXPERTS * 3; i++) {
      const d = tgt[i] - cur[i];
      if (Math.abs(d) > 0.0015) {
        cur[i] += d * k;
        moving = true;
      } else cur[i] = tgt[i];
    }
    if (moving) posAttr.needsUpdate = true;
    return moving;
  }

  function frame(now: number) {
    raf = 0;
    const dt = Math.min(0.05, last ? (now - last) / 1000 : 0.016);
    last = now;
    const moving = stepPositions(dt);
    if (auto && !dragging) group.rotation.y += 0.13 * dt;
    if (!dragging && Math.abs(velocity) > 0.01) {
      group.rotation.y += velocity * dt;
      velocity *= Math.exp(-dt * 2.6);
    } else if (!dragging) velocity = 0;
    renderer.render(scene, camera);
    if (active && (auto || moving || dragging || Math.abs(velocity) > 0.01)) schedule();
    else last = 0;
  }
  function schedule() {
    if (!raf && active) raf = requestAnimationFrame(frame);
  }

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // keep the whole column (and its offset) in view, also in a narrow frame
    const a = w / h;
    const need = Math.atan(5.8 / a / (2 * 12.4)) * 2 * (180 / Math.PI);
    camera.fov = Math.min(60, Math.max(26, need));
    camera.updateProjectionMatrix();
    material.uniforms.uScale.value = (h * dpr) / (2 * Math.tan((camera.fov * Math.PI) / 360));
    schedule();
    if (!auto) renderer.render(scene, camera);
  }

  const onLost = (e: Event) => {
    e.preventDefault();
    opts.onLost();
  };
  canvas.addEventListener("webglcontextlost", onLost);

  return {
    setState(v, c) {
      vram = v;
      capacity = c;
      retarget();
      schedule();
      if (!auto) renderer.render(scene, camera);
    },
    rotateBy(rad) {
      group.rotation.y += rad;
      schedule();
      if (!auto) renderer.render(scene, camera);
    },
    setDragging(on) {
      dragging = on;
      if (!on) schedule();
    },
    fling(v) {
      velocity = Math.max(-6, Math.min(6, v));
      schedule();
    },
    setActive(a) {
      active = a;
      if (a) schedule();
      else if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
        last = 0;
      }
    },
    resize,
    dispose() {
      active = false;
      if (raf) cancelAnimationFrame(raf);
      canvas.removeEventListener("webglcontextlost", onLost);
      geometry.dispose();
      material.dispose();
      ringMat.dispose();
      ringGeos.forEach((g) => g.dispose());
      renderer.dispose();
    },
  };
}
