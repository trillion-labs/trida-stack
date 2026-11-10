"""Build the single-file HTML report from a run's summary.json.  usage: build_report_html.py summary.json out.html"""
import json, sys

data = json.load(open(sys.argv[1]))
data["grid"] = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
data["vspec"] = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
data["sspec"] = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
data["fulleval"] = json.load(open(sys.argv[6])) if len(sys.argv) > 6 else None
data["dflash"] = json.load(open(sys.argv[7])) if len(sys.argv) > 7 else None
data["flare"] = json.load(open(sys.argv[8])) if len(sys.argv) > 8 else None
data["draftalign"] = json.load(open(sys.argv[9])) if len(sys.argv) > 9 else None
TEMPLATE = r"""<title>Trida bd4 Footprint</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#F1F4F7;--panel:#FFFFFF;--ink:#172029;--muted:#5C6B7A;--line:#D8DEE5;--soft:#E9EDF1;
 --vllm:#0E7C86;--vllm-soft:#D6EEF0;--sgl:#B85C2A;--sgl-soft:#F5E3D8;--ar:#6E7A87;--ar-soft:#E4E8EC;
 --good:#2E7D4F;--warn:#B7791F;--bad:#B23A3A;--accent:#0E7C86}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0F141A;--panel:#171E26;--ink:#E4E9EE;--muted:#9AA7B4;--line:#2A333D;--soft:#1F2830;
 --vllm:#3FB2BC;--vllm-soft:#12353A;--sgl:#E08A5A;--sgl-soft:#3D2418;--ar:#9AA7B4;--ar-soft:#263039;--good:#5CBF86;--warn:#E0A94D;--bad:#E06B6B;--accent:#3FB2BC}}
:root[data-theme="dark"]{--bg:#0F141A;--panel:#171E26;--ink:#E4E9EE;--muted:#9AA7B4;--line:#2A333D;--soft:#1F2830;
 --vllm:#3FB2BC;--vllm-soft:#12353A;--sgl:#E08A5A;--sgl-soft:#3D2418;--ar:#9AA7B4;--ar-soft:#263039;--good:#5CBF86;--warn:#E0A94D;--bad:#E06B6B;--accent:#3FB2BC}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;margin:0}
main{max-width:1040px;margin:0 auto;padding:36px 24px 72px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance}
h2{font-size:20px;font-weight:600;margin:44px 0 6px;letter-spacing:-.005em}
h3{font-size:14px;font-weight:600;margin:22px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
p{max-width:72ch;margin:6px 0}
.sub{color:var(--muted)}
.mono,td.n,th.n,.num{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.strip{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 8px}
.chip{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;background:var(--soft);font-size:13px;color:var(--muted)}
.chip .dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
.chip.done .dot{background:var(--good)} .chip.run .dot{background:var(--warn)}
.take{border-left:3px solid var(--accent);padding:6px 14px;margin:10px 0 18px;background:var(--panel);border-radius:0 8px 8px 0;max-width:none}
.take b{font-weight:600}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:28px;font-weight:500;margin-top:2px}
.card .v small{font-size:14px;color:var(--muted);font-weight:400;margin-left:4px}
.card .d{font-size:13px;color:var(--muted);margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.tw{overflow-x:auto;border-radius:10px}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;background:var(--soft)}
td.n,th.n{text-align:right} tr:last-child td{border-bottom:none}
.tag{display:inline-block;padding:1px 8px;border-radius:6px;font-size:12.5px;font-weight:500}
.tag.vllm{background:var(--vllm-soft);color:var(--vllm)} .tag.sgl{background:var(--sgl-soft);color:var(--sgl)} .tag.ar{background:var(--ar-soft);color:var(--ar)}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px 8px;margin:10px 0}
.chart svg{width:100%;height:auto;display:block;overflow:visible}
.chart .cap{font-size:13px;color:var(--muted);margin-top:6px}
svg text{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;fill:var(--muted)}
svg .ink{fill:var(--ink)} svg .axis{stroke:var(--line)} svg .grid{stroke:var(--line);stroke-dasharray:2 3}
.two{display:grid;gap:14px;grid-template-columns:1fr 1fr} @media(max-width:760px){.two{grid-template-columns:1fr}}
.lever{display:grid;grid-template-columns:1fr 3fr auto;gap:10px 14px;align-items:center;font-size:14px}
.lever .bar{height:14px;border-radius:4px;background:var(--vllm);position:relative}
.lever .bar.est{background:var(--vllm-soft);border:1px dashed var(--vllm)}
.lever .track{position:relative;height:14px;background:var(--soft);border-radius:4px}
.lever .par{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink);opacity:.6}
.foot{margin-top:40px;color:var(--muted);font-size:13px}
code{font-family:"IBM Plex Mono",monospace;font-size:13px;background:var(--soft);padding:1px 5px;border-radius:4px}
.pending{color:var(--muted);font-style:italic}
</style>
<main>
<h1>Trida bd4 Footprint</h1>
<p class="sub">Block-diffusion decode at block size 4, threshold 0.90, both engines, one instrumented run. GSM8K test set, greedy, thinking off, 1024-token cap.</p>
<div class="strip" id="strip"></div>
<div id="root"></div>
<div class="foot" id="foot"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const AR_MS = 4.8, CL = 4;
const $ = (h) => { const t = document.createElement('template'); t.innerHTML = h.trim(); return t.content; };
const root = document.getElementById('root');
const add = (h) => root.appendChild($(h));
const f1 = x => (Math.round(x*10)/10).toFixed(1), f2 = x => x.toFixed(2), f3 = x => x.toFixed(3), pc = x => f1(100*x)+'%';
const tagOf = (e, m) => m === 'causal' ? `<span class="tag ar">${e} · AR</span>` : `<span class="tag ${e==='vllm'?'vllm':'sgl'}">${e} · bd4</span>`;
const label = {
 'vllm-bd4-clean':'vLLM bd4', 'sglang-bd4-clean':'SGLang bd4 (reference)', 'vllm-causal-clean':'vLLM AR', 'sglang-causal-clean':'SGLang AR',
 'vllm-bd4-trace':'vLLM bd4 · trace server', 'vllm-bd4-timers':'vLLM bd4 · timer server', 'sweep':'concurrency sweep'};

// ---- status strip
const strip = document.getElementById('strip');
const allJobs = ['vllm-bd4-clean','sglang-bd4-clean','vllm-causal-clean','sglang-causal-clean','vllm-bd4-trace','vllm-bd4-timers','sweep'];
for (const j of allJobs) {
  const done = D.jobs_done.includes(j) || (j==='sweep' && D.sweep_done);
  strip.appendChild($(`<span class="chip ${done?'done':'run'}"><span class="dot"></span>${label[j]} · ${done?'done':'running'}</span>`));
}

// ---- derived numbers
const A = D.accuracy || {};
const tpfV = D.tok_per_fwd?.['vllm-bd4-trace']?.tok_per_fwd, tpfS = D.tok_per_fwd?.['sglang-bd4-clean']?.tok_per_fwd;
const ph = D.phase_ms_median, step = ph?.step, meanR = D.mean_rounds;
const speed = tpfV && step ? tpfV * AR_MS / step : null;
const bdTps = A['vllm-bd4-clean']?.per_gpu_tok_s ?? A['vllm-bd4-trace']?.per_gpu_tok_s, arTps = A['vllm-causal-clean']?.per_gpu_tok_s;

// ---- 0. the verdict
(function(){
  const sw = D.sweep || {}; const va = sw['vllm-causal'], vb = sw['vllm-bd4'];
  if (!va || !vb) return;
  const cs = Object.keys(va).filter(c => vb[c]).sort((a,b)=>a-b);
  const cells = cs.map(c => `<div class="card"><div class="k">concurrency ${c}</div><div class="v num">${f2(vb[c].tok_per_s/va[c].tok_per_s)}×</div><div class="d">bd4 ${Math.round(vb[c].tok_per_s)} vs AR ${Math.round(va[c].tok_per_s)} tok/s</div></div>`).join('');
  add(`<h2>0. The verdict so far</h2>
<p class="take"><b>On this checkpoint at block 4, diffusion is slower than AR at every concurrency, and batching widens the gap.</b> Both engines agree (vLLM bd4 = SGLang bd4 to within noise), so this is the model plus the diffusion step cost, not a port defect. AR amortizes one forward over the whole batch; diffusion does too, but each of its forwards yields only ~1.4 tokens per sequence at ~5× the cost. The engine work below (Fix B/C) removes the diffusion-specific overhead and reaches parity territory; beating AR needs a model that accepts ≥3 tokens per forward at bd4 accuracy, or a larger block that holds accuracy (block 32 currently does not: 57%).</p>
<div class="grid">${cells}</div>`);
})();

// ---- 1. the speed equation
add(`<h2>1. Where the speed goes</h2>
<p class="take">Speed-up over autoregressive decoding = <b>tokens per forward × (AR step time ÷ diffusion step time)</b>.
Right now vLLM bd4 gets <b>${tpfV?f2(tpfV):'—'} tokens per forward</b> but each step costs <b>${step?f1(step):'—'} ms against 4.8 ms for AR</b>, so it runs at
<b>${speed?f2(speed)+'×':'—'} AR speed</b>${bdTps&&arTps?` (measured end to end: ${f1(bdTps)} vs ${f1(arTps)} tok/s per GPU = ${f2(bdTps/arTps)}×)`:''}.
The acceptance side is healthy; the per-step cost is the problem.</p>
<div class="grid">
 <div class="card"><div class="k">tokens / forward, vLLM bd4</div><div class="v num">${tpfV?f3(tpfV):'—'}</div><div class="d">AR = 1.0 · ceiling at block 4 with the separate commit pass = 2.0 · ${D.trace_n_req||0} traced requests</div></div>
 <div class="card"><div class="k">tokens / forward, SGLang bd4</div><div class="v num">${tpfS?f3(tpfS):'<span class="pending">running</span>'}</div><div class="d">from the server's own forward counters</div></div>
 <div class="card"><div class="k">step time, vLLM bd4</div><div class="v num">${step?f1(step):'—'}<small>ms</small></div><div class="d">median, CUDA-synced · AR step 4.8 ms · break-even needs ${step?f1(step/AR_MS):'—'} tok/fwd</div></div>
 <div class="card"><div class="k">resulting speed vs AR</div><div class="v num">${speed?f2(speed)+'×':'—'}</div><div class="d">parity = 1.0×</div></div>
</div>`);

// lever ladder (estimates derived from the measured numbers)
if (tpfV && step && ph && meanR) {
  const fixC_tpf = CL / meanR;                       // commit fused into next denoise: forwards/block = rounds
  const fixB_step = step - ph.snap - Math.max(ph.gdn - 1.0, 0); // drop snapshot/restore + host overhead of the 24-layer GDN override (kernel ~1 ms stays)
  const rows = [
    ['today', tpfV*AR_MS/step, false, `${f2(tpfV)} tok/fwd · ${f1(step)} ms`],
    ['Fix C: fuse commit into next denoise', fixC_tpf*AR_MS/step, true, `${f2(fixC_tpf)} tok/fwd · ${f1(step)} ms`],
    ['Fix B: fuse GDN override + drop snapshot/restore', tpfV*AR_MS/fixB_step, true, `${f2(tpfV)} tok/fwd · ${f1(fixB_step)} ms`],
    ['B + C', fixC_tpf*AR_MS/fixB_step, true, `${f2(fixC_tpf)} tok/fwd · ${f1(fixB_step)} ms`],
    ['B + C + every block in one round (upper bound)', CL*AR_MS/fixB_step, true, `4.00 tok/fwd · ${f1(fixB_step)} ms`],
  ];
  const mx = Math.max(1.05, ...rows.map(r => r[1]));
  add(`<h3>lever ladder · speed vs AR</h3><div class="chart"><div class="lever">${rows.map(r =>
    `<div>${r[0]}</div><div class="track"><div class="bar ${r[2]?'est':''}" style="width:${100*r[1]/mx}%"></div><div class="par" style="left:${100/mx}%"></div></div><div class="num">${f2(r[1])}× <span class="sub" style="font-size:12px">${r[3]}</span></div>`).join('')}</div>
  <div class="cap">Solid = measured. Dashed = first-order estimate from the measured rounds-per-block and phase profile (Fix B assumes the GDN override's host overhead and the per-step snapshot/restore go away, keeping ~1 ms of kernel). The vertical mark is AR parity.</div></div>`);
}

// ---- 2. accuracy & throughput
add(`<h2>2. Accuracy and throughput</h2>
<p class="take">${A['vllm-bd4-clean'] && A['sglang-bd4-clean'] ?
 `On the full 1319 items vLLM bd4 scores <b>${pc(A['vllm-bd4-clean'].acc)}</b> vs SGLang bd4 <b>${pc(A['sglang-bd4-clean'].acc)}</b> (±${f1(100*A['vllm-bd4-clean'].ci)} pts each), with AR at ${pc(A['vllm-causal-clean'].acc)} / ${pc(A['sglang-causal-clean'].acc)}.` :
 `AR baselines are in on the full set (vLLM ${A['vllm-causal-clean']?pc(A['vllm-causal-clean'].acc):'—'}, SGLang ${A['sglang-causal-clean']?pc(A['sglang-causal-clean'].acc):'—'}); the two bd4 full-set jobs are still running, so the bd4 rows below are the 200- and 50-item diagnostic servers.`}
 Numbers use the fork's GSM8K client (boxed-answer prompt), which reads a few points lower than our earlier "####" harness for every engine alike.</p>
<div class="tw"><table><thead><tr><th>engine</th><th class="n">items</th><th class="n">accuracy</th><th class="n">±95%</th><th class="n">tok/s per GPU</th><th class="n">avg tokens</th><th class="n">hit cap</th><th class="n">errors</th><th class="n">replicas</th></tr></thead><tbody>
${['vllm-bd4-clean','sglang-bd4-clean','vllm-causal-clean','sglang-causal-clean','vllm-bd4-trace','vllm-bd4-timers'].map(j => { const r = A[j]; if (!r) return `<tr><td>${label[j]}</td><td colspan="8" class="pending">running</td></tr>`;
 return `<tr><td>${label[j]} ${tagOf(r.engine,r.mode)}</td><td class="n">${r.n}</td><td class="n"><b>${pc(r.acc)}</b></td><td class="n">${f1(100*r.ci)}</td><td class="n">${f1(r.per_gpu_tok_s)}${r.trace!=='none'?'<span class="sub">†</span>':''}</td><td class="n">${Math.round(r.avg_tok)}</td><td class="n">${r.hit_max}</td><td class="n">${r.errors}</td><td class="n">${r.nrep}</td></tr>`;}).join('')}
</tbody></table></div>
<p class="sub" style="font-size:13px">tok/s per GPU = completion tokens ÷ wall ÷ replicas with one request in flight per replica (single-stream). † trace/timer servers carry diagnostic host syncs; use the clean rows for speed.</p>`);

// ---- 3. acceptance
if (D.rounds_hist) {
  const rh = D.rounds_hist, tot = Object.values(rh).reduce((a,b)=>a+b,0), ks = Object.keys(rh).map(Number).sort();
  const pf = D.pos_first || {};
  const W=460,H=190,pad=36; const bw = (W-pad*2)/ks.length*0.6;
  const bars = ks.map((k,i)=>{const v=rh[k]/tot, x=pad+(W-pad*2)*(i+0.5)/ks.length-bw/2, h=(H-50)*v;
    return `<rect x="${x}" y="${H-30-h}" width="${bw}" height="${h}" rx="3" fill="var(--vllm)"/><text x="${x+bw/2}" y="${H-30-h-6}" text-anchor="middle" class="ink">${f1(100*v)}%</text><text x="${x+bw/2}" y="${H-12}" text-anchor="middle">${k} round${k>1?'s':''}</text>`;}).join('');
  const pks = Object.keys(pf).sort(); const pw=(W-pad*2)/pks.length*0.6;
  const pbars = pks.map((k,i)=>{const v=pf[k][1]/pf[k][0], x=pad+(W-pad*2)*(i+0.5)/pks.length-pw/2, h=(H-50)*v;
    return `<rect x="${x}" y="${H-30-h}" width="${pw}" height="${h}" rx="3" fill="var(--vllm)"/><text x="${x+pw/2}" y="${H-30-h-6}" text-anchor="middle" class="ink">${f1(100*v)}%</text><text x="${x+pw/2}" y="${H-12}" text-anchor="middle">slot ${k}</text>`;}).join('');
  add(`<h2>3. Acceptance: how many rounds a block takes</h2>
<p class="take">A block of 4 has one carried seed and 3 masked slots. <b>${pc(rh[1]/tot)} of blocks close in a single round</b>; the mean is <b>${f2(meanR)} rounds</b> plus one commit forward, which is exactly where ${f3(CL/(meanR+1))} tokens per forward comes from. Slot 1 (right after the seed) is accepted first-round ${pc(pf['1'][1]/pf['1'][0])} of the time; slot 3 only ${pc(pf['3'][1]/pf['3'][0])}. Positions far from the seed are the bottleneck, not the seed itself.</p>
<div class="two">
 <div class="chart"><svg viewBox="0 0 ${W} ${H}"><line x1="${pad}" y1="${H-30}" x2="${W-pad}" y2="${H-30}" class="axis"/>${bars}</svg><div class="cap">Denoise rounds per block, ${tot.toLocaleString()} blocks from ${D.trace_n_req} requests.</div></div>
 <div class="chart"><svg viewBox="0 0 ${W} ${H}"><line x1="${pad}" y1="${H-30}" x2="${W-pad}" y2="${H-30}" class="axis"/>${pbars}</svg><div class="cap">Share of blocks where the slot committed in the first round.</div></div>
</div>
<p class="sub" style="font-size:13px">Seed-chain check: ${D.seed_mismatch?D.seed_mismatch[0]:'—'} of ${D.seed_mismatch?D.seed_mismatch[1].toLocaleString():'—'} block boundaries had a seed that differed from the next block's slot 0 (expected 0 after the group-mapping fix).</p>`);
}

// ---- 4. calibration & knobs
if (D.calibration) {
  const cal = Object.entries(D.calibration).map(([k,v])=>[parseFloat(k),v[0],v[1]]).sort((a,b)=>a[0]-b[0]);
  const W=940,H=250,pl=46,pr=16,pt=14,pb=40; const xs = x => pl+(W-pl-pr)*(x/0.9), ys = y => pt+(H-pt-pb)*(1-y);
  const maxN = Math.max(...cal.map(c=>c[1]));
  const bw = (W-pl-pr)/18*0.9;
  const barsN = cal.map(c=>`<rect x="${xs(c[0])+1}" y="${ys(0.0)-(H-pt-pb)*0.35*c[1]/maxN}" width="${bw-2}" height="${(H-pt-pb)*0.35*c[1]/maxN}" fill="var(--soft)" stroke="var(--line)"/>`).join('');
  const pts = cal.map(c=>`${xs(c[0]+0.025)},${ys(c[2]/c[1])}`).join(' ');
  const dots = cal.map(c=>`<circle cx="${xs(c[0]+0.025)}" cy="${ys(c[2]/c[1])}" r="3.5" fill="${c[2]/c[1]>0.9?'var(--warn)':'var(--vllm)'}"/>`).join('');
  const grid = [0.25,0.5,0.75,1].map(g=>`<line x1="${pl}" x2="${W-pr}" y1="${ys(g)}" y2="${ys(g)}" class="grid"/><text x="${pl-6}" y="${ys(g)+4}" text-anchor="end">${Math.round(100*g)}%</text>`).join('');
  const xt = [0.1,0.3,0.5,0.7,0.9].map(t=>`<text x="${xs(t)}" y="${H-pb+16}" text-anchor="middle">${t.toFixed(1)}</text>`).join('');
  const K = D.knobs || {};
  add(`<h2>4. Is the 0.90 gate too strict?</h2>
<p class="take">Every rejected prediction was checked against the token that finally landed in that slot. <b>Predictions rejected at confidence 0.80–0.90 were right ${pc((K['thr0.80']?.[1]||0)/(K['thr0.80']?.[0]||1))} of the time</b>; the curve is monotone and well calibrated. Lowering the gate to 0.80 would have committed ${(K['thr0.80']?.[0]||0).toLocaleString()} more slots early (${pc((K['thr0.80']?.[0]||0)/(K['thr0.80']?.[2]||1))} of all rejections) at ${pc((K['thr0.80']?.[1]||0)/(K['thr0.80']?.[0]||1))} precision. These are first-order estimates: committing earlier changes what the model sees next, so the real effect must be measured, and the threshold stays at 0.90 until that measurement is made.</p>
<div class="chart"><svg viewBox="0 0 ${W} ${H}">${grid}${barsN}<line x1="${xs(0.9)}" x2="${xs(0.9)}" y1="${pt}" y2="${H-pb}" stroke="var(--bad)" stroke-dasharray="4 3"/><text x="${xs(0.9)-4}" y="${pt+10}" text-anchor="end" fill="var(--bad)">gate 0.90</text>
<polyline points="${pts}" fill="none" stroke="var(--vllm)" stroke-width="2"/>${dots}${xt}<text x="${(pl+W-pr)/2}" y="${H-6}" text-anchor="middle">confidence of the rejected prediction</text></svg>
<div class="cap">Line: share of rejected predictions that matched the final token, by confidence bin (amber = above 90%). Grey bars: how many rejections fell in each bin (${cal.reduce((a,c)=>a+c[1],0).toLocaleString()} total). Decisions overall: ${Object.entries(D.types||{}).map(([k,v])=>`${k} ${v.toLocaleString()}`).join(' · ')}.</div></div>
<h3>gate variants, simulated offline on the same trace</h3>
<div class="tw"><table><thead><tr><th>variant</th><th class="n">extra early commits</th><th class="n">of which correct</th><th class="n">precision</th><th class="n">rejections recovered</th></tr></thead><tbody>
${Object.entries(K).map(([k,v])=>`<tr><td>${k}</td><td class="n">${v[0].toLocaleString()}</td><td class="n">${v[1].toLocaleString()}</td><td class="n"><b>${pc(v[1]/Math.max(v[0],1))}</b></td><td class="n">${pc(v[0]/Math.max(v[2],1))}</td></tr>`).join('')}
</tbody></table></div>`);
}

// ---- 5. per-step cost
if (ph) {
  const segs = [['snapshot / restore', ph.snap, 'var(--sgl)'], ['GDN override (24 layers, host-bound)', ph.gdn, 'var(--vllm)'], ['rest of forward (attention, MLP, glue)', Math.max(ph.fwd-ph.gdn,0), 'var(--ar)'], ['sampler / gate', ph.samp, 'var(--warn)']];
  const tot = segs.reduce((a,s)=>a+s[1],0); let x=0;
  const W=940,H=70;
  const rects = segs.map(s=>{const w=(W)*s[1]/tot; const r=`<rect x="${x}" y="10" width="${w}" height="26" fill="${s[2]}"/><text x="${x+w/2}" y="58" text-anchor="middle">${f1(s[1])} ms</text>`; x+=w; return r;}).join('');
  add(`<h2>5. What a decode step costs</h2>
<p class="take">Median step <b>${f1(ph.step)} ms</b> (${D.timer_steps?.toLocaleString()} steps, CUDA-synced), against 4.8 ms for one AR token. <b>${f1(ph.snap+ph.gdn)} ms of it is not model compute</b>: ${f1(ph.snap)} ms restoring GDN state before every step and ${f1(ph.gdn)} ms of host-side dispatch in the eager per-layer GDN override (its kernel work is about 1 ms). That is Fix B's target. The unsynced trace server saw a median host step of ${D.step_ms_median?f1(D.step_ms_median):'—'} ms and p90 ${D.step_ms_p90?f1(D.step_ms_p90):'—'} ms.</p>
<div class="chart"><svg viewBox="0 0 ${W} ${H}">${rects}</svg><div class="cap">${segs.map(s=>`<span style="display:inline-block;width:10px;height:10px;background:${s[2]};border-radius:2px;margin:0 6px 0 10px"></span>${s[0]}`).join('')}</div></div>`);
}

// ---- 6. sweep
add(`<h2>6. Serving throughput vs concurrency</h2>` + (D.sweep ? (()=>{
  const S = D.sweep; const cs = [...new Set(Object.values(S).flatMap(v=>Object.keys(v).map(Number)))].sort((a,b)=>a-b);
  return `<p class="take">Fixed 512-token outputs, 64 prompts, strict concurrency waves. Aggregate tokens per second per GPU as concurrency rises.</p>
<div class="tw"><table><thead><tr><th>engine</th>${cs.map(c=>`<th class="n">C=${c}</th>`).join('')}</tr></thead><tbody>
${Object.entries(S).map(([n,v])=>`<tr><td>${n}</td>${cs.map(c=>`<td class="n">${v[c]?(v[c].n_ok?Math.round(v[c].tok_per_s)+' tok/s':'<span class="pending">error</span>'):'—'}</td>`).join('')}</tr>`).join('')}
</tbody></table></div>`;})() : `<p class="pending">Queued behind the full-set jobs; this section fills in when it completes.</p>`));

// ---- 7. threshold / block-size grid
(function(){
  const G = D.grid; if (!G) { add(`<h2>7. Threshold and block-size grid</h2><p class="pending">Grid run (thr 0.80 × block 4/32, control b32/0.90) not yet available.</p>`); return; }
  const cfgs = [
    ['b4 · thr 0.90 (baseline)', A['vllm-bd4-clean'], A['sglang-bd4-clean'], D.tok_per_fwd?.['vllm-bd4-trace']?.tok_per_fwd, D.tok_per_fwd?.['sglang-bd4-clean']?.tok_per_fwd, D.traces?.['vllm-bd4-trace']],
    ['b4 · thr 0.80', G.accuracy?.['vllm-b4-t080'], G.accuracy?.['sglang-b4-t080'], G.tok_per_fwd?.['vllm-b4-t080']?.tok_per_fwd, G.tok_per_fwd?.['sglang-b4-t080']?.tok_per_fwd, G.traces?.['vllm-b4-t080']],
    ['b32 · thr 0.80', G.accuracy?.['vllm-b32-t080'], G.accuracy?.['sglang-b32-t080'], G.tok_per_fwd?.['vllm-b32-t080']?.tok_per_fwd, G.tok_per_fwd?.['sglang-b32-t080']?.tok_per_fwd, G.traces?.['vllm-b32-t080']],
    ['b32 · thr 0.90 (control)', G.accuracy?.['vllm-b32-t090'], G.accuracy?.['sglang-b32-t090'], G.tok_per_fwd?.['vllm-b32-t090']?.tok_per_fwd, G.tok_per_fwd?.['sglang-b32-t090']?.tok_per_fwd, G.traces?.['vllm-b32-t090']],
  ];
  const arAcc = A['vllm-causal-clean']?.acc;
  const cell = (r) => r ? `<b>${pc(r.acc)}</b> <span class="sub">±${f1(100*r.ci)}</span>` : '<span class="pending">running</span>';
  const rows = cfgs.map(c => `<tr><td>${c[0]}</td><td class="n">${cell(c[1])}</td><td class="n">${cell(c[2])}</td><td class="n">${c[3]?f3(c[3]):'—'}</td><td class="n">${c[4]?f3(c[4]):'—'}</td><td class="n">${c[5]?f2(c[5].mean_rounds):'—'}</td><td class="n">${c[1]?c[1].hit_max:'—'} / ${c[2]?c[2].hit_max:'—'}</td></tr>`).join('');
  const done = cfgs.filter(c=>c[1]&&c[2]);
  let take = `Same client, same full 1319 items, AR reference ${arAcc?pc(arAcc):'—'}. `;
  if (G.accuracy?.['vllm-b4-t080'] && A['vllm-bd4-clean']) {
    const d = G.accuracy['vllm-b4-t080'].acc - A['vllm-bd4-clean'].acc, t = (G.tok_per_fwd?.['vllm-b4-t080']?.tok_per_fwd||0)/(D.tok_per_fwd?.['vllm-bd4-trace']?.tok_per_fwd||1);
    take += `Dropping the gate to 0.80 at block 4 moves vLLM accuracy by <b>${d>=0?'+':''}${f1(100*d)} pts</b> and tokens per forward by <b>${t?('×'+f2(t)):'—'}</b>. `;
  }
  if (G.accuracy?.['vllm-b32-t080'] && G.accuracy?.['vllm-b4-t080']) take += `Block 32 at the same gate scores <b>${pc(G.accuracy['vllm-b32-t080'].acc)}</b> vs block 4's ${pc(G.accuracy['vllm-b4-t080'].acc)}. `;
  if (!done.length) take += `Jobs are running; rows fill in as they land.`;
  add(`<h2>7. Threshold and block-size grid</h2><p class="take">${take}</p>
  <div class="tw"><table><thead><tr><th>config</th><th class="n">vLLM acc</th><th class="n">SGLang acc</th><th class="n">vLLM tok/fwd</th><th class="n">SGLang tok/fwd</th><th class="n">vLLM rounds/block</th><th class="n">hit cap (v / s)</th></tr></thead><tbody>${rows}</tbody></table></div>`);
  // first-round acceptance vs slot for each vLLM trace
  const tr = cfgs.filter(c=>c[5]&&c[5].pos_first).map(c=>[c[0], c[5]]);
  if (tr.length) {
    const W=940,H=260,pl=46,pr=16,pt=14,pb=40; const maxSlot = Math.max(...tr.map(t=>Math.max(...Object.keys(t[1].pos_first).map(Number))));
    const xs = x => pl+(W-pl-pr)*((x-1)/Math.max(maxSlot-1,1)), ys = y => pt+(H-pt-pb)*(1-y);
    const cols = ['var(--vllm)','var(--sgl)','var(--warn)','var(--ar)'];
    const lines = tr.map((t,i)=>{ const pts = Object.entries(t[1].pos_first).map(([k,v])=>[Number(k), v[1]/v[0]]).sort((a,b)=>a[0]-b[0]);
      return `<polyline points="${pts.map(p=>`${xs(p[0])},${ys(p[1])}`).join(' ')}" fill="none" stroke="${cols[i%4]}" stroke-width="2"/>` + pts.map(p=>`<circle cx="${xs(p[0])}" cy="${ys(p[1])}" r="2.5" fill="${cols[i%4]}"/>`).join(''); }).join('');
    const grid = [0.25,0.5,0.75,1].map(g=>`<line x1="${pl}" x2="${W-pr}" y1="${ys(g)}" y2="${ys(g)}" class="grid"/><text x="${pl-6}" y="${ys(g)+4}" text-anchor="end">${Math.round(100*g)}%</text>`).join('');
    const xt = Array.from({length:maxSlot},(_, i)=>i+1).filter(x=>x===1||x%4===0||x===maxSlot).map(x=>`<text x="${xs(x)}" y="${H-pb+16}" text-anchor="middle">${x}</text>`).join('');
    add(`<div class="chart"><svg viewBox="0 0 ${W} ${H}">${grid}${lines}${xt}<text x="${(pl+W-pr)/2}" y="${H-6}" text-anchor="middle">canvas slot (distance from the carried seed)</text></svg>
    <div class="cap">Share of blocks where the slot committed in the first denoise round. ${tr.map((t,i)=>`<span style="display:inline-block;width:10px;height:10px;background:${cols[i%4]};border-radius:2px;margin:0 6px 0 10px"></span>${t[0]}`).join('')}</div></div>`);
  }
})();

// ---- 8. self-spec (AR-Trust) on vLLM
(function(){
  const S = D.vspec; if (!S) { add(`<h2>8. AR-Trust (self-spec) on vLLM</h2><p class="pending">not yet available</p>`); return; }
  const A2 = S.accuracy || {}, I = S.identity_vs_ref || {}, T = S.tok_per_fwd || {}, TR = S.traces || {};
  const STAGE = {vspec:'v1 port', s2:'Fix S2 (fused GDN)', s3:'S3', s4:'S4 (spec-shape, FULL cuda graph)'};
  const rows = Object.keys(A2).filter(k => /^(vspec|s2|s3|s4)-n\d+-(200|full)/.test(k)).sort().map(k => {
    const a = A2[k], id = I[k], t = T[k], tr = TR[k]; const N = /n(\d+)/.exec(k)[1]; const stg = STAGE[/^([a-z0-9]+)-/.exec(k)[1]] || k;
    const hist = tr?.spec_accept_hist ? Object.entries(tr.spec_accept_hist).sort((x,y)=>x[0]-y[0]).map(([kk,v],_,arr)=>`${kk}:${f1(100*v/arr.reduce((s,e)=>s+e[1],0))}%`).join(' ') : '—';
    return `<tr><td>N=${N} (canvas ${2*N-1}) · ${stg}</td><td class="n">${a.n}</td><td class="n"><b>${pc(a.acc)}</b></td><td class="n">${id?pc(id.ref_acc):'—'}</td><td class="n">${id?`${id.identical}/${id.n} (${pc(id.identical/id.n)})`:'—'}</td><td class="n"><b>${t?f3(t.tok_per_fwd):'—'}</b></td><td class="n">${hist}</td></tr>`; }).join('');
  const sw = S.sweep || {}, ar = D.sweep?.['vllm-causal'] || {}, bd = D.sweep?.['vllm-bd4'] || {};
  const cs = ['1','4','8','16'];
  const ssw = D.sspec?.sweep || {};
  const sw2 = S.sweep_sweep_s2 || {}; const sw4 = S.sweep_sweep_s4 || {}; const ar4 = sw4['vllm-causal'];
  const swRows = [['vLLM AR (reference)', ar], ['vLLM diffusion bd4', bd], ...Object.entries(sw).filter(([k])=>k.startsWith('vllm-spec')).map(([k,v])=>['vLLM self-spec '+k.replace('vllm-spec-','')+' (v1 port)', v]),
     ...Object.entries(sw2).filter(([k])=>k.startsWith('vllm-spec')).map(([k,v])=>['vLLM self-spec '+k.replace('vllm-spec-','')+' (Fix S2, fused GDN)', v]),
     ...(ar4?[['vLLM AR (same job as S4 rows)', ar4]]:[]),
     ...Object.entries(sw4).filter(([k])=>k.startsWith('vllm-spec')).map(([k,v])=>['<b>vLLM self-spec '+k.replace('vllm-spec-','')+' (S4: FULL cuda graph)</b>', v]),
     ...(ssw['sglang-causal']?[['SGLang AR', ssw['sglang-causal']]]:[]), ...Object.entries(ssw).filter(([k])=>k.startsWith('sglang-spec')).map(([k,v])=>['SGLang self-spec '+k.replace('sglang-spec-',''), v])]
    .map(([n,v]) => `<tr><td>${n}</td>${cs.map(c => `<td class="n">${v[c]?Math.round(v[c].tok_per_s):'—'}</td>`).join('')}${cs.map(c => `<td class="n">${v[c]&&ar[c]?f2(v[c].tok_per_s/ar[c].tok_per_s)+'×':'—'}</td>`).join('')}</tr>`).join('');
  const n4 = (S.sweep_sweep_s2||{})['vllm-spec-n4'] || sw['vllm-spec-n4'];
  const s4n4 = sw4['vllm-spec-n4'], arRef = ar4 && ar4['1'] ? ar4 : ar;
  const s4take = s4n4 ? `<br><b>Update (S4, 2026-09-10):</b> the step is now presented to vLLM as a spec-decode batch (bonus token + 2N−2 drafts) so FULL cuda graphs capture the whole forward: step cost 20.8 → 16.9 → 11.6 → 10.0 → <b>7.5 ms</b> (v1 → S → S2 → S3 → S4), output still token-identical to the earlier stages. Self-spec N=4 is now <b>${cs.map(c=>s4n4[c]&&arRef[c]?`${f2(s4n4[c].tok_per_s/arRef[c].tok_per_s)}× at C=${c}`:null).filter(Boolean).join(', ')}</b> of vLLM AR.` : '';
  add(`<h2>8. AR-Trust (self-spec) on vLLM</h2>
<p class="take">FLARE's fast mode, ported today: the model drafts N−1 tokens from MASK slots and the same forward verifies them against exact AR logits, so the output is AR-greedy up to bf16 near-ties. <b>Accuracy matches AR within a point, ${I['vspec-n4-200']?pc(I['vspec-n4-200'].identical/I['vspec-n4-200'].n):'—'} of generations are token-identical, and it yields ${T['vspec-n4-200']?f2(T['vspec-n4-200'].tok_per_fwd):'—'} tokens per forward.</b> That makes it ${n4&&bd['1']?f1(n4['1'].tok_per_s/bd['1'].tok_per_s)+'×':'—'} faster than the diffusion path at C=1 and ${n4&&bd['16']?f1(n4['16'].tok_per_s/bd['16'].tok_per_s)+'×':'—'} at C=16, but still ${n4&&ar['1']?f2(n4['1'].tok_per_s/ar['1'].tok_per_s)+'×':'—'} of AR: the step is a 2N−1-token forward through the same eager GDN path plus a naive per-layer state commit, so the per-step cost is again what stands between the measured tokens-per-forward and a win over AR.</p>
<div class="tw"><table><thead><tr><th>config</th><th class="n">items</th><th class="n">accuracy</th><th class="n">AR on same items</th><th class="n">identical to AR greedy</th><th class="n">tok/fwd</th><th class="n">accepted specs / forward</th></tr></thead><tbody>${rows}</tbody></table></div>
${(()=>{ const G2 = D.sspec; if (!G2) return ''; const A3 = G2.accuracy||{}, T3 = G2.tok_per_fwd||{}; const arS = A['sglang-causal-clean'];
  const r = Object.keys(A3).sort().map(k => { const a=A3[k], t=T3[k]; const g=/g(\d+)/.exec(k)?.[1]; return `<tr><td>SGLang self-spec g${g} (block ${2*g-1})</td><td class="n">${a.n}</td><td class="n"><b>${pc(a.acc)}</b></td><td class="n">${arS?pc(arS.acc):'—'}</td><td class="n">${f1(a.per_gpu_tok_s)}</td><td class="n">${arS?f1(arS.per_gpu_tok_s):'—'}</td><td class="n"><b>${t?f3(t.tok_per_fwd):'—'}</b></td></tr>`; }).join('');
  const FL = D.flare||{}, FA = FL.accuracy||{}, FT = FL.tok_per_fwd||{}, flAR = FA['flare4b-causal'];
  const rf = Object.keys(FA).filter(k=>k.startsWith('flare4b-spec')).sort().map(k => { const a=FA[k], t=FT[k]; const g=/g(\d+)/.exec(k)?.[1]; return `<tr><td><b>FLARE-4B (public checkpoint)</b> AR-Trust g${g} (block ${2*g-1})</td><td class="n">${a.n}</td><td class="n"><b>${pc(a.acc)}</b></td><td class="n">${flAR?pc(flAR.acc):'—'}</td><td class="n">${f1(a.per_gpu_tok_s)}</td><td class="n">${flAR?f1(flAR.per_gpu_tok_s):'—'}</td><td class="n"><b>${t?f3(t.tok_per_fwd):'—'}</b></td></tr>`; }).join('');
  return `<h3>the reference engine, same mode, full 1319 items</h3><div class="tw"><table><thead><tr><th>config</th><th class="n">items</th><th class="n">accuracy</th><th class="n">own AR</th><th class="n">tok/s per GPU</th><th class="n">own AR tok/s</th><th class="n">tok/fwd</th></tr></thead><tbody>${r}${rf}</tbody></table></div>
${rf?`<p class="sub" style="font-size:13px"><b>FLARE-4B rows (2026-09-11):</b> the authors' released 4B checkpoint (block 4, AR weight 1.0, random-mask objective, ~9.4B tokens) on the same SGLang reference and protocol (greedy, no-think, 400 GSM8K items). Its AR-Trust accepts 3.38 tokens per forward at g4 vs trida's 2.74 — the recipe buys ~23% acceptance; DFlash's separate drafter gets 6.5. This is the bar the draft-aligned fine-tune has to clear.</p>`:''}
<p class="sub" style="font-size:13px">On SGLang, AR-Trust beats its own AR single-stream (163 vs 121 tok/s at g8), which is the paper's claim reproduced. On vLLM the same mode is at 0.5× of vLLM AR because vLLM AR is much faster (219 tok/s) and our self-spec step is not yet fused.</p>`; })()}
${s4take?`<p class="take">${s4take.replace('<br>','')}</p>`:''}
<h3>serving throughput (fixed 512-token outputs, 64 prompts, one replica)</h3>
<div class="tw"><table><thead><tr><th>engine</th>${cs.map(c=>`<th class="n">C=${c} tok/s</th>`).join('')}${cs.map(c=>`<th class="n">C=${c} vs AR</th>`).join('')}</tr></thead><tbody>${swRows}</tbody></table></div>
<p class="sub" style="font-size:13px">Attention on vLLM is fully causal over the canvas (FA3 has no intra-block custom mask); the reference lets MASK rows see the whole block, so its drafts are stronger. Verification is exact either way. GDN state after each step was fingerprinted against a fresh prefill of the emitted prefix: 0.3–2% relative difference, flat over 65 steps (numerics, no drift).</p>`);
})();


// ---- 9. full eval: GSM8K + FunctionChat + Ko-AgentBench x {AR, self-spec N=4/8/32}
(function(){
  const F = D.fulleval; if (!F) return;
  const MODES = [['causal','vLLM AR (greedy)'],['selfspec-n4','self-spec N=4'],['selfspec-n8','self-spec N=8'],['selfspec-n32','self-spec N=32']];
  const g = F.gsm8k||{}, fc = F.fc||{}, ko = F.koab||{};
  const rows = MODES.map(([m,n]) => { const a=g[m], f=fc[m], k=ko[m];
    return `<tr><td>${n}</td><td class="n">${a?`<b>${pc(a.acc)}</b> <span class="sub">${a.correct}/${a.total}</span>`:'—'}</td><td class="n">${a?Math.round(a.tok_s_16gpu):'—'}</td>
      <td class="n">${f?.singlecall_score?.total_pass_rate!=null?f3(f.singlecall_score.total_pass_rate):'—'}</td><td class="n">${f?.dialog_score?.['avg(micro)']!=null?f3(f.dialog_score['avg(micro)']):'—'}</td><td class="n">${f?.calldecision_score?.total_pass_rate!=null?f3(f.calldecision_score.total_pass_rate):'—'}</td>
      <td class="n">${k?.sr_weighted!=null?`<b>${f3(k.sr_weighted)}</b>`:'—'}</td><td class="n">${k?.tps_mean!=null?Math.round(k.tps_mean):'—'}</td></tr>`; }).join('');
  const lv = ['L1','L2','L3','L4','L5','L6','L7'];
  const krows = MODES.map(([m,n]) => { const k=ko[m]; if(!k||!k.levels) return ''; return `<tr><td>${n}</td>${lv.map(l=>`<td class="n">${k.levels[l]?f2(k.levels[l].SR):'—'}</td>`).join('')}<td class="n">${k.api_timeouts??'—'} / ${k.context_exceeded??'—'}</td></tr>`; }).join('');
  const g0=g['causal'], g4=g['selfspec-n4'];
  add(`<h2>9. Full eval on vLLM: AR vs self-spec N=4 / 8 / 32 (greedy, thinking on)</h2>
<p class="take">Same checkpoint, same vLLM build, greedy decoding with thinking enabled, one sequence per replica. <b>Self-spec tracks AR on every benchmark within the run-to-run noise of thinking-mode generations</b>: GSM8K ${g4&&g0?`${pc(g4.acc)} vs ${pc(g0.acc)}`:'—'}, FunctionChat within ~1–2 points on all three subsets, Ko-AgentBench task-weighted success rate ${ko['selfspec-n4']&&ko['causal']?`${f3(ko['selfspec-n4'].sr_weighted)} vs ${f3(ko['causal'].sr_weighted)}`:'—'} (91 tasks, so ±0.05 is noise). N=4 and N=8 run GSM8K at ${g4&&g0?f2(g4.tok_s_16gpu/g0.tok_s_16gpu)+'×':'—'} AR throughput on 16 replicas; N=32 pays for its 63-token forward and lands at AR speed. Token-level identity to AR drops with thinking on (long traces accumulate bf16 near-tie flips), which is why per-benchmark scores move by a point or two in both directions.</p>
<div class="tw"><table><thead><tr><th>mode</th><th class="n">GSM8K acc</th><th class="n">GSM8K tok/s (16 GPU)</th><th class="n">FunctionChat singlecall</th><th class="n">FC dialog avg(micro)</th><th class="n">FC calldecision</th><th class="n">Ko-AgentBench SR (task-weighted)</th><th class="n">KoAB tok/s per call</th></tr></thead><tbody>${rows}</tbody></table></div>
<h3>Ko-AgentBench success rate by level</h3>
<div class="tw"><table><thead><tr><th>mode</th>${lv.map(l=>`<th class="n">${l}</th>`).join('')}<th class="n">LLM timeouts / context-window errors</th></tr></thead><tbody>${krows}</tbody></table></div>
<p class="sub" style="font-size:13px">Protocol: GSM8K boxed prompt, max_tokens 8192, 12–13% of items hit the cap in every mode (thinking loops). FunctionChat v2.0.0 with gpt-4.1 judge, temperature 0 (harness knob), three subsets run concurrently. Ko-AgentBench L1–L7, concurrency 8, cache-mode read, gpt-4.1-mini judge, per-call LLM timeout 1200 s (the harness default of 60 s truncated the first AR run and was rerun); the same ~28 context-window errors (32k) occur in every mode. Levels have 10–20 tasks each. FunctionChat wall time is 2.5× longer for self-spec because vLLM disables prefix caching for the diffusion plugin; Ko-AgentBench per-call throughput is 1.3–2× AR.</p>`);
})();


// ---- 10. external baseline: DFlash (block-diffusion drafter) vs self-spec vs AR, same GPU / protocol
(function(){
  const X = D.dflash; if (!X) return;
  const label = {ar:'AR (stock Qwen3.5-4B)', dflash4:'DFlash block 4', dflash8:'DFlash block 8', dflash16:'DFlash block 16', selfspec4:'trida self-spec N=4', selfspec8:'trida self-spec N=8', selfspec16:'trida self-spec N=16', selfspec32:'trida self-spec N=32'};
  const order = ['ar','dflash4','dflash8','dflash16','selfspec4','selfspec8','selfspec16','selfspec32'];
  const cs=['1','4','8','16'];
  let html = `<h2>10. External baseline: DFlash vs self-spec vs AR</h2>
<p class="take">DFlash pairs the stock Qwen3.5-4B with a separate 6-layer block-diffusion drafter (<code>z-lab/Qwen3.5-4B-DFlash</code>) and verifies a whole block per step; our self-spec drafts from the same weights with no extra parameters. Same protocol as section 8 (64 GSM8K prompts, 512 fixed output tokens, greedy, one H100 per server). AR speed is identical for both targets (217 tok/s at C=1), so the speedups are directly comparable even though the targets differ. <b>DFlash block 16 accepts ~6.5 of 16 drafted tokens per step and reaches 2.4× AR at C=1; self-spec N=4 accepts ~1.3 of 3 and reaches 1.6×.</b> Both converge toward AR as concurrency fills the GPU (1.56× vs 1.03× at C=16). Wider self-spec blocks are slower on vLLM: the 2N−1-row forward outgrows the acceptance gain.</p>`;
  for (const [job, J] of Object.entries(X.jobs||{})) {
    const R = J.results||{}, A = J.accept_len_serverlog||{}, ar = R.ar||{};
    const engine = job.includes('sglang') ? 'SGLang' : 'vLLM';
    const names = [...new Set([...order.filter(n=>R[n]), ...Object.keys(R).filter(n=>!order.includes(n))])];
    const rows = names.map(n => { const v = R[n]||{};
      const cells = cs.map(c => { const r=v[c]; if(!r) return '<td class="n">—</td>'; const sp = (n!=='ar' && ar[c]) ? ` <span class="sub">${f2(r.tok_per_s/ar[c].tok_per_s)}×</span>` : ''; const err = r.n_err ? ` <span class="sub">err ${r.n_err}</span>` : ''; return `<td class="n">${Math.round(r.tok_per_s)}${sp}${err}</td>`; }).join('');
      const acc = A[n] ?? cs.map(c=>v[c]?.accept).find(x=>x);
      return `<tr><td>${label[n]||n}</td>${cells}<td class="n">${acc?f2(acc):(n.startsWith('selfspec')?'2.3–2.5 tok/fwd':'—')}</td></tr>`; }).join('');
    html += `<h3>${engine} (${job})</h3><div class="tw"><table><thead><tr><th>config</th>${cs.map(c=>`<th class="n">C=${c} tok/s</th>`).join('')}<th class="n">accept len / tok per fwd</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  html += `<p class="sub" style="font-size:13px">DFlash acceptance lengths are vLLM's own SpecDecoding metrics (mean accepted tokens per verify step, drafted block included). Self-spec N=16/32 were run with 4 sequences per server: their state ring (24 layers × 2·R·N MB) does not fit beside the KV cache at 16 sequences. N≥8 self-spec at C≥8 hits the known mixed prompt+canvas batch limitation.</p>`;
  add(html);
})();

// ---- 11. draft-aligned fine-tune pilot (+ FLARE-4B public checkpoint under our protocol)
(function(){
  const FL = D.flare; if (FL && FL.accuracy) {
    const A5 = FL.accuracy, T5 = FL.tok_per_fwd||{};
    const rows = Object.keys(A5).sort().map(k => { const a=A5[k], t=T5[k]; const g=/g(\d+)/.exec(k)?.[1];
      return `<tr><td>${g?`FLARE-4B self-spec g${g} (block ${2*g-1})`:'FLARE-4B AR (causal)'}</td><td class="n">${a.n}</td><td class="n"><b>${pc(a.acc)}</b></td><td class="n">${t?.tok_per_fwd?f2(t.tok_per_fwd):'—'}</td><td class="n">${a.per_gpu_tok_s?Math.round(a.per_gpu_tok_s):'—'}</td></tr>`; }).join('');
    add(`<h2>11a. Reference: the public FLARE-4B checkpoint under our protocol</h2>
<p class="take">Same SGLang fork, same GSM8K protocol (400 items, greedy). FLARE-4B accepts <b>3.38</b> tokens per forward at g4 where trida (step_18000) accepts 2.74 on the same fork and 2.3–2.4 on vLLM. That gap is the drafter's training, not the engine, and is what the fine-tune below tried to close.</p>
<div class="tw"><table><thead><tr><th>config</th><th class="n">items</th><th class="n">accuracy</th><th class="n">tok / fwd</th><th class="n">tok/s per GPU</th></tr></thead><tbody>${rows}</tbody></table></div>`);
  }
  const P = D.draftalign; if (!P) return;
  const rows = P.rows.map(r => `<tr><td>${r.ckpt}</td><td class="n">N=${r.N}</td><td class="n"><b>${f2(r.tok_fwd)}</b></td><td class="n">${f1(r.cold)}%</td><td class="n">${f1(r.slot1)}%</td><td class="n">${f1(r.allacc)}%</td><td class="n">${r.acc}</td></tr>`).join('');
  const L = P.loss||{}; const lossRow = (name, xs) => `<tr><td>${name}</td>${xs.map(([st,v])=>`<td class="n">${f2(v)}</td>`).join('')}</tr>`;
  add(`<h2>11. Draft-aligned fine-tune: 100-step pilot (${P.date})</h2>
<p class="take"><b>Verdict: ${P.verdict}.</b> The noisy stream was fine-tuned on its own greedy outputs with the decoder's canvas mask pattern (${P.recipe}); data: ${P.data}. Training loss fell (diffusion 1.42 → 1.25, AR 0.19 → 0.08) but acceptance under the real self-spec decode did not move: N=4 drifts up 3% inside run-to-run noise, N=8 is flat. The AR guard held (${P.ar_guard.step_18000} → ${P.ar_guard.step_50} → ${P.ar_guard.step_100} on the same 30 items).</p>
<div class="tw"><table><thead><tr><th>checkpoint</th><th class="n">mode</th><th class="n">tok / fwd</th><th class="n">cold steps</th><th class="n">warm: slot-1 accepted</th><th class="n">warm: all accepted</th><th class="n">GSM8K (30)</th></tr></thead><tbody>${rows}</tbody></table></div>
<p class="sub" style="font-size:13px">A <b>cold</b> step follows any rejection: the canvas is one clean token plus MASKs, there is nothing to verify, exactly one token is emitted. Warm steps carry N−1 specs. tok/fwd is therefore set by the warm all-accepted rate (49.5% at N=4, 10% at N=8), and slot-1 acceptance is only ~84% even though that slot sees exactly the AR head's context — the number a working drafter fine-tune must move first.</p>
<h3>training loss (every 10 steps)</h3>
<div class="tw"><table><thead><tr><th>loss</th>${(L.diff||[]).map(([st])=>`<th class="n">${st}</th>`).join('')}</tr></thead><tbody>${lossRow('diffusion (noisy stream, canvas masks)', L.diff||[])}${lossRow('AR (clean stream, self-targets)', L.ar||[])}</tbody></table></div>
<p class="sub" style="font-size:13px"><b>What the pilot does and does not say.</b> 100 steps ≈ 40M supervised tokens at LR 5e-6 is a small nudge; the loss was still falling at step 100. But the loss drop did not reach acceptance, so before spending a day on generation + 300 steps the next check is an offline per-slot diagnostic: run step_18000 and step_100 through the trainer's forward with the exact inference canvas and measure slot-by-slot agreement with the AR head. If offline slot-1 agreement is far above the online 84%, the ceiling is inference fidelity (vLLM runs the canvas with causal attention; training is bidirectional within the block) and more training will not help; if it is also ~84%, it is capacity, and a separate drafter is the route. Domain gap noted: training data is agentic multi-turn, this eval is GSM8K.</p>
<p class="sub" style="font-size:13px"><b>Infra learned on the way:</b> ${P.infra}</p>`);
})();

// ---- foot
document.getElementById('foot').innerHTML = `Run <code>${D.run}</code> · generated ${D.generated} · code md5 <code>${(D.manifest?.code_md5||'').slice(0,8)}</code> · vLLM: canvas 4, threshold 0.90, max 8 denoise rounds, PIECEWISE cuda-graph (S4 self-spec rows: FULL_AND_PIECEWISE), one sequence per replica · SGLang: block_size 3 (= block 4 with carried seed), threshold 0.90, greedy · same checkpoint (step_18000) · raw files: <code>&lt;run&gt;/&lt;job&gt;/gsm8k_details_*.json</code>, <code>trace_rep*.jsonl</code>, <code>REPORT.md</code>, <code>summary.json</code>.`;
</script>
"""
html = TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
open(sys.argv[2], "w").write(html)
print("wrote", sys.argv[2], len(html), "bytes")
