/* Kinematics Lab — resolved-rate (velocity-level) control of planar arms.
 *
 * All modes share one controller:
 *   q̇ = J#·v + (I − J#·J)·q̇0,   v = ẋ_d + K·(x_d − x)
 *   J# = Jᵀ(J·Jᵀ + λ²I)⁻¹        (λ = 0 → pseudo-inverse, λ > 0 → damped least squares)
 * Joint rates are scaled down together if any exceeds the joint speed limit,
 * and joint angles are clamped to their limits after integration.
 * World units are meters, y points up; the canvas flips y when drawing.
 */
(function(){
  const root = document.getElementById('lab');
  if (!root) return;

  const $ = id => document.getElementById(id);
  const canvas = $('labCanvas'), ctx = canvas.getContext('2d');
  const speedCanvas = $('labSpeedPlot'), errCanvas = $('labErrPlot');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const C = {
    text: '#1c2330', faint: '#666d7c', border: '#e2e4e9', grid: '#eef0f3',
    teal: '#2f6b63', tealLine: '#3f7c74', navy: '#1f3a5f', amber: '#b7791f', warn: '#9a4515',
    okFill: '#d6e7e3', badFill: '#f1d6a8'
  };
  const SERIES = [C.teal, C.navy, C.amber];
  const D2R = Math.PI / 180;
  const G = 9.81;
  const QD_MAX = 6;          // joint speed limit, rad/s
  const LAMBDA = 0.12;       // DLS damping, m
  const PLOT_N = 360;        // samples kept in the strip charts

  // ---------- Arms ----------
  const ARMS = {
    two: {
      L: [1.10, 0.95], m: [4, 3],
      lim: [[-45 * D2R, 225 * D2R], [-150 * D2R, 150 * D2R]],
      q0: [60 * D2R, -70 * D2R]
    },
    three: {
      L: [0.85, 0.70, 0.50],
      lim: [[-45 * D2R, 225 * D2R], [-160 * D2R, 160 * D2R], [-160 * D2R, 160 * D2R]],
      q0: [70 * D2R, -60 * D2R, -40 * D2R]
    }
  };
  const TAU_MAX = [110, 45]; // N·m ratings for the 2-link torque check

  // ---------- State ----------
  const state = {
    mode: 'two',
    dls: true,
    ellipse: true,
    payload: 3,
    objective: 'obstacle',
    gain: 6, speed: 1, ff: true, reach: 0.55,
    q: ARMS.two.q0.slice(),
    target: { x: 1.2, y: 1.0 },
    obstacle: { x: 0.70, y: 0.95, r: 0.20 },
    t: 0, paused: reduceMotion,
    idle: true, idleT: 0,
    qdHist: [], errHist: [],
    lastQd: [0, 0, 0], lastErr: 0,
    collision: false, satur: false
  };

  // ---------- Kinematics ----------
  function fk(L, q){
    const pts = [{ x: 0, y: 0 }];
    let phi = 0, x = 0, y = 0;
    for (let i = 0; i < L.length; i++){
      phi += q[i]; x += L[i] * Math.cos(phi); y += L[i] * Math.sin(phi);
      pts.push({ x, y });
    }
    return pts;
  }
  // 2×n Jacobian, returned as [row0[], row1[]]
  function jac(L, q){
    const n = L.length, r0 = new Array(n).fill(0), r1 = new Array(n).fill(0);
    const phis = []; let phi = 0;
    for (let i = 0; i < n; i++){ phi += q[i]; phis.push(phi); }
    for (let j = 0; j < n; j++){
      for (let k = j; k < n; k++){
        r0[j] += -L[k] * Math.sin(phis[k]);
        r1[j] +=  L[k] * Math.cos(phis[k]);
      }
    }
    return [r0, r1];
  }
  // J# = Jᵀ (J Jᵀ + λ² I)⁻¹  → n×2
  function jinv(J, lambda){
    const n = J[0].length;
    let a = 0, b = 0, c = 0;
    for (let i = 0; i < n; i++){ a += J[0][i] * J[0][i]; b += J[0][i] * J[1][i]; c += J[1][i] * J[1][i]; }
    a += lambda * lambda; c += lambda * lambda;
    let det = a * c - b * b;
    if (Math.abs(det) < 1e-9) det = det < 0 ? -1e-9 : 1e-9;
    const i00 = c / det, i01 = -b / det, i11 = a / det;
    const out = [];
    for (let i = 0; i < n; i++){
      out.push([J[0][i] * i00 + J[1][i] * i01, J[0][i] * i01 + J[1][i] * i11]);
    }
    return out;
  }
  // Velocity manipulability ellipse: singular values / directions of J (via eig of J Jᵀ)
  function ellipseOf(J){
    const n = J[0].length;
    let a = 0, b = 0, c = 0;
    for (let i = 0; i < n; i++){ a += J[0][i] * J[0][i]; b += J[0][i] * J[1][i]; c += J[1][i] * J[1][i]; }
    const m = (a + c) / 2, d = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    return { s1: Math.sqrt(Math.max(m + d, 0)), s2: Math.sqrt(Math.max(m - d, 0)), ang: 0.5 * Math.atan2(2 * b, a - c), w: Math.sqrt(Math.max(a * c - b * b, 0)) };
  }
  function gravityTorque(q, payload){
    const [L1, L2] = ARMS.two.L, [m1, m2] = ARMS.two.m;
    const c1 = Math.cos(q[0]), c12 = Math.cos(q[0] + q[1]);
    const t2 = G * (m2 * L2 / 2 * c12 + payload * L2 * c12);
    const t1 = G * (m1 * L1 / 2 * c1 + m2 * (L1 * c1 + L2 / 2 * c12) + payload * (L1 * c1 + L2 * c12)) ;
    return [t1, t2];
  }

  // ---------- Null-space objectives (3-link) ----------
  function obstacleCost(q){
    const L = ARMS.three.L, pts = fk(L, q), ob = state.obstacle;
    const infl = 0.30; let H = 0;
    for (let i = 0; i < L.length; i++){
      // Skip the outer half of the last link so the cost never fights the tip reaching its target.
      const sMax = i === L.length - 1 ? 3 : 6;
      for (let s = 0; s <= sMax; s++){
        const t = s / 6;
        const x = pts[i].x + (pts[i + 1].x - pts[i].x) * t, y = pts[i].y + (pts[i + 1].y - pts[i].y) * t;
        const d = Math.hypot(x - ob.x, y - ob.y) - ob.r;
        if (d < infl) H += (infl - d) ** 2;
      }
    }
    return H;
  }
  function nullGradient(q){
    const n = q.length;
    if (state.objective === 'none') return new Array(n).fill(0);
    if (state.objective === 'center'){
      // Keep joints 2..n near the middle of their range (avoids limits and awkward folds).
      return q.map((qi, i) => {
        if (i === 0) return 0;
        const [lo, hi] = ARMS.three.lim[i], mid = (lo + hi) / 2, span = hi - lo;
        return -2.5 * (qi - mid) / (span * span) * 4;
      });
    }
    // Obstacle: numerical gradient of the clearance cost, descended.
    const g = [], h = 1e-4, H0 = obstacleCost(q);
    for (let i = 0; i < n; i++){
      const qp = q.slice(); qp[i] += h;
      g.push(-(obstacleCost(qp) - H0) / h * 25);
    }
    return g;
  }

  // ---------- Trajectory (figure-8) ----------
  function pathAt(t){
    // Far end of the figure-8 sits at distance D from the base; the "reach" slider moves D toward full stretch.
    const D = 1.45 + state.reach * (2.08 - 1.45);
    const cy = 0.9, a = 0.5, b = 0.32;
    const cx = Math.sqrt(Math.max(D * D - cy * cy, 0)) - a;
    const w = 2 * Math.PI / 7 * state.speed;
    return {
      x: cx + a * Math.sin(w * t), y: cy + b * Math.sin(2 * w * t),
      vx: a * w * Math.cos(w * t), vy: 2 * b * w * Math.cos(2 * w * t)
    };
  }

  // ---------- Controller step ----------
  function step(dt){
    const arm = state.mode === 'three' ? ARMS.three : ARMS.two;
    const L = arm.L, q = state.q, n = L.length;
    const pts = fk(L, q), tip = pts[n];

    let xd, vff = { x: 0, y: 0 }, K;
    if (state.mode === 'track'){
      state.t += dt;
      const p = pathAt(state.t);
      xd = p; if (state.ff){ vff = { x: p.vx, y: p.vy }; }
      K = state.gain;
    } else {
      xd = state.target; K = 4;
    }
    const ex = xd.x - tip.x, ey = xd.y - tip.y;
    let vx = vff.x + K * ex, vy = vff.y + K * ey;
    const vmax = state.mode === 'track' ? 4 : 1.6, vn = Math.hypot(vx, vy);
    if (vn > vmax){ vx *= vmax / vn; vy *= vmax / vn; }

    const J = jac(L, q), Ji = jinv(J, state.dls ? LAMBDA : 0);
    const qd = new Array(n).fill(0);
    for (let i = 0; i < n; i++) qd[i] = Ji[i][0] * vx + Ji[i][1] * vy;

    const peakOf = v => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
    let peak = peakOf(qd);
    state.satur = peak > QD_MAX;
    if (state.satur){
      // Task alone exceeds the joint speed limit: scale it down, no room left for the secondary objective.
      for (let i = 0; i < n; i++) qd[i] *= QD_MAX / peak;
    } else if (n > 2){
      // Secondary objective in the null space N = I − J#·J, using only the speed headroom the task leaves.
      const g = nullGradient(q), nq = new Array(n).fill(0);
      for (let i = 0; i < n; i++){
        for (let j = 0; j < n; j++){
          const N = (i === j ? 1 : 0) - (Ji[i][0] * J[0][j] + Ji[i][1] * J[1][j]);
          nq[i] += N * g[j];
        }
      }
      let alpha = 1;
      for (let it = 0; it < 20 && peakOf(qd.map((v, i) => v + alpha * nq[i])) > QD_MAX; it++) alpha *= 0.7;
      for (let i = 0; i < n; i++) qd[i] += alpha * nq[i];
    }

    for (let i = 0; i < n; i++){
      q[i] += qd[i] * dt;
      q[i] = Math.max(arm.lim[i][0], Math.min(arm.lim[i][1], q[i]));
    }
    state.lastQd = qd;
    state.lastErr = Math.hypot(ex, ey);
    state.xd = xd;
  }

  // ---------- Rendering ----------
  let W = 0, H = 0, dpr = 1, scale = 1, ox = 0, oy = 0;
  const VIEW = { x0: -2.25, x1: 2.25, y0: -1.0, y1: 2.25 };
  const toPx = p => ({ x: ox + p.x * scale, y: oy - p.y * scale });
  const toWorld = (px, py) => ({ x: (px - ox) / scale, y: (oy - py) / scale });

  let wsCanvas = null, wsKey = '';
  function buildWorkspace(){
    const key = W + 'x' + H + ':' + state.payload;
    if (key === wsKey) return;
    wsKey = key;
    wsCanvas = document.createElement('canvas');
    wsCanvas.width = Math.round(W * dpr); wsCanvas.height = Math.round(H * dpr);
    const w = wsCanvas.getContext('2d');
    w.setTransform(dpr, 0, 0, dpr, 0, 0);
    const [l1, l2] = ARMS.two.lim, step = 1.2 * D2R, okPts = [], badPts = [];
    for (let a = l1[0]; a <= l1[1]; a += step){
      for (let b = l2[0]; b <= l2[1]; b += step){
        const p = fk(ARMS.two.L, [a, b])[2];
        const t = gravityTorque([a, b], state.payload);
        (Math.abs(t[0]) <= TAU_MAX[0] && Math.abs(t[1]) <= TAU_MAX[1] ? okPts : badPts).push(toPx(p));
      }
    }
    // Any configuration that can hold the pose wins, so draw infeasible first.
    const sz = Math.max(1.5, scale * 0.022);
    w.fillStyle = C.badFill; badPts.forEach(p => w.fillRect(p.x - sz / 2, p.y - sz / 2, sz, sz));
    w.fillStyle = C.okFill; okPts.forEach(p => w.fillRect(p.x - sz / 2, p.y - sz / 2, sz, sz));
  }

  function resize(){
    const r = canvas.getBoundingClientRect();
    if (!r.width) return;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = r.width; H = r.width * (VIEW.y1 - VIEW.y0) / (VIEW.x1 - VIEW.x0);
    canvas.style.height = H + 'px';
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    scale = W / (VIEW.x1 - VIEW.x0); ox = -VIEW.x0 * scale; oy = VIEW.y1 * scale;
    [speedCanvas, errCanvas].forEach(c => {
      const cr = c.getBoundingClientRect();
      c.width = Math.round(cr.width * dpr); c.height = Math.round(cr.height * dpr);
    });
    wsKey = '';
  }

  function line(a, b, color, width, dash){
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.setLineDash([]);
  }
  function dot(p, r, fill, stroke){
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
    if (fill){ ctx.fillStyle = fill; ctx.fill(); }
    if (stroke){ ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
  }

  function draw(){
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.lineCap = 'round';
    const arm = state.mode === 'three' ? ARMS.three : ARMS.two;
    const reach = arm.L.reduce((s, v) => s + v, 0);
    const base = toPx({ x: 0, y: 0 });

    if (state.mode === 'two'){ buildWorkspace(); ctx.drawImage(wsCanvas, 0, 0, W, H); }

    // grid
    ctx.strokeStyle = 'rgba(28,35,48,0.06)'; ctx.lineWidth = 1;
    for (let gx = Math.ceil(VIEW.x0 * 2) / 2; gx <= VIEW.x1; gx += 0.5){ const p = toPx({ x: gx, y: 0 }); ctx.beginPath(); ctx.moveTo(p.x, 0); ctx.lineTo(p.x, H); ctx.stroke(); }
    for (let gy = Math.ceil(VIEW.y0 * 2) / 2; gy <= VIEW.y1; gy += 0.5){ const p = toPx({ x: 0, y: gy }); ctx.beginPath(); ctx.moveTo(0, p.y); ctx.lineTo(W, p.y); ctx.stroke(); }

    ctx.beginPath(); ctx.arc(base.x, base.y, reach * scale, 0, 2 * Math.PI);
    ctx.strokeStyle = '#d5d8df'; ctx.setLineDash([3, 5]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);

    // scale bar
    const sb0 = { x: 12, y: H - 14 }, sb1 = { x: 12 + 0.5 * scale, y: H - 14 };
    line(sb0, sb1, C.faint, 2);
    ctx.fillStyle = C.faint; ctx.font = '12px "IBM Plex Mono", monospace'; ctx.fillText('0.5 m', sb1.x + 6, sb1.y + 4);

    // path / obstacle
    if (state.mode === 'track'){
      ctx.beginPath();
      const T = 7 / state.speed;
      for (let i = 0; i <= 160; i++){ const p = toPx(pathAt(T * i / 160)); i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); }
      ctx.strokeStyle = 'rgba(31,58,95,0.35)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
    }
    if (state.mode === 'three'){
      const o = toPx(state.obstacle);
      dot(o, state.obstacle.r * scale, 'rgba(154,69,21,0.12)', C.warn);
      ctx.fillStyle = C.warn; ctx.fillText('obstacle', o.x - 30, o.y + 4);
    }

    // arm
    const pts = fk(arm.L, state.q).map(toPx);
    ctx.fillStyle = '#f3f4f6'; ctx.strokeStyle = '#c9cdd6'; ctx.lineWidth = 1;
    ctx.fillRect(base.x - 16, base.y - 6, 32, 30); ctx.strokeRect(base.x - 16, base.y - 6, 32, 30);
    const linkColors = [C.tealLine, C.navy, C.amber];
    state.collision = false;
    if (state.mode === 'three'){
      const wp = fk(arm.L, state.q), ob = state.obstacle;
      for (let i = 0; i < arm.L.length; i++){
        let hit = false;
        for (let s = 0; s <= 10; s++){
          const t = s / 10, x = wp[i].x + (wp[i + 1].x - wp[i].x) * t, y = wp[i].y + (wp[i + 1].y - wp[i].y) * t;
          if (Math.hypot(x - ob.x, y - ob.y) < ob.r) hit = true;
        }
        if (hit) state.collision = true;
        line(pts[i], pts[i + 1], hit ? C.warn : linkColors[i], 7);
      }
    } else {
      for (let i = 0; i < arm.L.length; i++) line(pts[i], pts[i + 1], linkColors[i], 7);
    }
    for (let i = 0; i < pts.length - 1; i++) dot(pts[i], i ? 5 : 6, C.text);
    const tip = pts[pts.length - 1];
    dot(tip, 7, null, C.navy);

    // manipulability ellipse at the tip
    if (state.ellipse){
      const e = ellipseOf(jac(arm.L, state.q)), k = 0.28 * scale;
      ctx.save(); ctx.translate(tip.x, tip.y); ctx.rotate(-e.ang);
      ctx.beginPath(); ctx.ellipse(0, 0, Math.max(e.s1 * k, 0.5), Math.max(e.s2 * k, 0.5), 0, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(31,58,95,0.08)'; ctx.fill();
      ctx.strokeStyle = 'rgba(31,58,95,0.6)'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.restore();
    }

    // target
    if (state.xd){
      const t = toPx(state.xd);
      const unreachable = Math.hypot(state.xd.x, state.xd.y) > reach;
      line({ x: t.x - 6, y: t.y }, { x: t.x + 6, y: t.y }, unreachable ? C.warn : C.teal, 2);
      line({ x: t.x, y: t.y - 6 }, { x: t.x, y: t.y + 6 }, unreachable ? C.warn : C.teal, 2);
    }
  }

  function drawPlot(c, series, opts){
    const g = c.getContext('2d'), w = c.width / dpr, h = c.height / dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const pad = { l: 40, r: 8, t: 8, b: 8 }, pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    const y = v => pad.t + ph * (1 - (v - opts.min) / (opts.max - opts.min));
    g.font = '11px "IBM Plex Mono", monospace'; g.fillStyle = C.faint; g.textAlign = 'right';
    opts.ticks.forEach(tv => {
      g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath(); g.moveTo(pad.l, y(tv)); g.lineTo(w - pad.r, y(tv)); g.stroke();
      g.fillText(String(tv), pad.l - 6, y(tv) + 4);
    });
    (opts.limits || []).forEach(lv => {
      g.strokeStyle = C.warn; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(pad.l, y(lv)); g.lineTo(w - pad.r, y(lv)); g.stroke(); g.setLineDash([]);
    });
    series.forEach((s, si) => {
      g.strokeStyle = s.color; g.lineWidth = 1.6; g.beginPath();
      s.data.forEach((v, i) => {
        const px = pad.l + pw * (i + PLOT_N - s.data.length) / (PLOT_N - 1);
        const py = y(Math.max(opts.min, Math.min(opts.max, v)));
        i ? g.lineTo(px, py) : g.moveTo(px, py);
      });
      g.stroke();
    });
  }

  // ---------- Readouts ----------
  const out = {
    w: $('labManip'), err: $('labErr'), qd: $('labQd'), flag: $('labFlag'),
    t1: $('labTau1'), t2: $('labTau2'), b1: $('labTau1Bar'), b2: $('labTau2Bar')
  };
  let lastText = 0;
  function readouts(now){
    if (now - lastText < 100) return;
    lastText = now;
    const arm = state.mode === 'three' ? ARMS.three : ARMS.two;
    const e = ellipseOf(jac(arm.L, state.q));
    out.w.textContent = e.w.toFixed(2);
    out.err.textContent = (state.lastErr * 1000).toFixed(0);
    out.qd.textContent = Math.max(...state.lastQd.map(Math.abs)).toFixed(1);
    const flags = [];
    if (e.w < 0.15) flags.push('near singularity');
    if (state.satur) flags.push('joint speed limit hit');
    if (state.mode === 'three' && state.collision) flags.push('collision');
    out.flag.textContent = flags.join(' · ');
    out.flag.hidden = !flags.length;
    if (state.mode === 'two'){
      const t = gravityTorque(state.q, state.payload);
      [[t[0], out.t1, out.b1, TAU_MAX[0]], [t[1], out.t2, out.b2, TAU_MAX[1]]].forEach(([v, tx, bar, max]) => {
        tx.textContent = Math.abs(v).toFixed(0) + ' / ' + max + ' N·m';
        bar.style.width = Math.min(Math.abs(v) / max / 1.5, 1) * 100 + '%';
        bar.classList.toggle('over', Math.abs(v) > max);
      });
    }
  }

  // ---------- Loop ----------
  let rafId = null, lastT = null, visible = false;
  function loop(now){
    const dt = lastT ? Math.min((now - lastT) / 1000, 0.05) : 0;
    lastT = now;
    if (state.idle && state.mode !== 'track' && !reduceMotion){
      state.idleT += dt;
      const t = state.idleT;
      if (state.mode === 'three'){
        // Sweep behind the obstacle, where the elbow's posture decides whether the arm collides.
        state.target = { x: 0.55 + 0.9 * Math.sin(t * 0.5), y: 1.65 };
      } else {
        const r = 2.05 * (0.62 + 0.36 * Math.sin(t * 0.45));
        const a = (90 + 70 * Math.sin(t * 0.28)) * D2R;
        state.target = { x: r * Math.cos(a), y: r * Math.sin(a) };
      }
    }
    if (dt > 0 && !(state.mode === 'track' && state.paused)){
      const sub = 4;
      for (let i = 0; i < sub; i++) step(dt / sub);
      state.qdHist.push(state.lastQd.slice()); if (state.qdHist.length > PLOT_N) state.qdHist.shift();
      state.errHist.push(state.lastErr * 1000); if (state.errHist.length > PLOT_N) state.errHist.shift();
    }
    draw();
    const n = state.mode === 'three' ? 3 : 2;
    const series = [];
    for (let j = 0; j < n; j++) series.push({ color: SERIES[j], data: state.qdHist.map(r => r[j] || 0) });
    drawPlot(speedCanvas, series, { min: -8, max: 8, ticks: [-6, 0, 6], limits: [-QD_MAX, QD_MAX] });
    const emax = state.mode === 'track' ? 200 : 1000;
    drawPlot(errCanvas, [{ color: C.navy, data: state.errHist }], { min: 0, max: emax, ticks: [0, emax / 2, emax] });
    readouts(now);
    rafId = visible && !document.hidden ? requestAnimationFrame(loop) : null;
  }
  function start(){ if (rafId === null && visible && !document.hidden){ lastT = null; rafId = requestAnimationFrame(loop); } }
  new IntersectionObserver(es => { visible = es[0].isIntersecting; start(); }).observe(canvas);
  document.addEventListener('visibilitychange', start);
  if ('ResizeObserver' in window) new ResizeObserver(() => { resize(); }).observe(canvas.parentElement);
  resize();

  // ---------- Input ----------
  let idleTimer = null, dragging = null;
  function wake(){ state.idle = false; clearTimeout(idleTimer); }
  function sleepSoon(){ clearTimeout(idleTimer); idleTimer = setTimeout(() => { state.idle = true; }, 3000); }
  function evWorld(e){ const r = canvas.getBoundingClientRect(); return toWorld(e.clientX - r.left, e.clientY - r.top); }
  canvas.addEventListener('pointerdown', e => {
    if (state.mode === 'track') return;
    const p = evWorld(e); wake();
    if (state.mode === 'three' && Math.hypot(p.x - state.obstacle.x, p.y - state.obstacle.y) < state.obstacle.r + 0.08){
      dragging = 'obstacle'; canvas.setPointerCapture(e.pointerId);
    } else { dragging = 'target'; state.target = p; }
  });
  canvas.addEventListener('pointermove', e => {
    if (state.mode === 'track') return;
    const p = evWorld(e);
    if (dragging === 'obstacle'){ state.obstacle.x = p.x; state.obstacle.y = p.y; return; }
    if (e.pointerType === 'mouse' || dragging){ wake(); state.target = p; }
  });
  ['pointerup', 'pointercancel'].forEach(t => canvas.addEventListener(t, () => { dragging = null; if (window.matchMedia('(pointer: coarse)').matches) sleepSoon(); }));
  canvas.addEventListener('pointerleave', () => { if (dragging !== 'obstacle') dragging = null; sleepSoon(); });
  canvas.addEventListener('keydown', e => {
    if (state.mode === 'track') return;
    const s = e.shiftKey ? 0.15 : 0.05;
    const mv = { ArrowLeft: [-s, 0], ArrowRight: [s, 0], ArrowUp: [0, s], ArrowDown: [0, -s] }[e.key];
    if (!mv) return;
    e.preventDefault(); wake();
    state.target = { x: state.target.x + mv[0], y: state.target.y + mv[1] };
  });
  canvas.addEventListener('blur', sleepSoon);

  // ---------- Controls ----------
  const DESCRIPTIONS = {
    two: 'A 2-link arm driven by resolved-rate (velocity) control toward the crosshair. The shaded region is everything the arm can reach within its joint limits: teal where the motors can hold the pose against gravity, amber where the payload would exceed a joint’s torque rating. Raise the payload and watch the amber band grow.',
    three: 'With three joints and a 2-D target, the arm has one redundant degree of freedom. The task uses J#, and a secondary objective moves the arm in the null space (I − J#J), changing its posture without moving the tip. This is the same idea I used on Canadarm2, where secondary objectives like time and energy picked among the 7-joint solutions. Drag the obstacle; switch the objective to None and the elbow will happily pass through it.',
    track: 'The arm tracks a figure-8 using feedforward plus proportional feedback: q̇ = J#(ẋ_d + K·e). Lower the gain or turn off feedforward to see the tracking error grow. Push the path toward full stretch: with the plain pseudo-inverse, joint speeds spike and saturate near the singularity, while damped least squares trades a little tracking error for bounded joint speeds.'
  };
  const tabs = Array.from(root.querySelectorAll('[role="tab"]'));
  function setMode(mode, focus){
    state.mode = mode;
    tabs.forEach(t => {
      const on = t.dataset.mode === mode;
      t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    root.querySelectorAll('[data-for]').forEach(p => { p.hidden = !p.dataset.for.split(' ').includes(mode); });
    $('labDesc').textContent = DESCRIPTIONS[mode];
    state.q = (mode === 'three' ? ARMS.three : ARMS.two).q0.slice();
    state.qdHist = []; state.errHist = []; state.t = 0; state.idle = true;
    canvas.setAttribute('aria-label', {
      two: 'Two-link arm with shaded reachable workspace and gravity torque limits. Move the pointer, or focus and use arrow keys, to move the target.',
      three: 'Three-link redundant arm avoiding a draggable obstacle using null-space control. Move the pointer, or focus and use arrow keys, to move the target.',
      track: 'Two-link arm tracking a figure-eight trajectory.'
    }[mode]);
    canvas.style.cursor = mode === 'track' ? 'default' : 'crosshair';
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => setMode(t.dataset.mode));
    t.addEventListener('keydown', e => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (!d) return;
      e.preventDefault();
      setMode(tabs[(i + d + tabs.length) % tabs.length].dataset.mode, true);
    });
  });

  function seg(groupId, key, parse){
    const btns = Array.from($(groupId).querySelectorAll('button'));
    btns.forEach(b => b.addEventListener('click', () => {
      state[key] = parse(b.dataset.value);
      btns.forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    }));
  }
  seg('labInverse', 'dls', v => v === 'dls');
  seg('labObjective', 'objective', v => v);
  $('labEllipse').addEventListener('change', e => { state.ellipse = e.target.checked; });
  const play = $('labPlay');
  function syncPlay(){ play.textContent = state.paused ? 'Play' : 'Pause'; play.setAttribute('aria-pressed', String(!state.paused)); }
  play.addEventListener('click', () => { state.paused = !state.paused; syncPlay(); });
  syncPlay();
  $('labFF').addEventListener('change', e => { state.ff = e.target.checked; });
  function slider(id, key, fmt){
    const el = $(id), o = $(id + 'Out');
    const upd = () => { state[key] = parseFloat(el.value); o.textContent = fmt(state[key]); };
    el.addEventListener('input', upd); upd();
  }
  slider('labPayload', 'payload', v => v.toFixed(1) + ' kg');
  slider('labGain', 'gain', v => v.toFixed(0) + ' s⁻¹');
  slider('labSpeed', 'speed', v => v.toFixed(1) + '×');
  slider('labReach', 'reach', v => Math.round(v * 100) + '%');

  setMode('two');
  start();
})();
