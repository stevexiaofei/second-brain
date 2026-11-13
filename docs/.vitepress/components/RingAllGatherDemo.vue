<template>
  <section class="ring-demo" aria-label="Ring all-gather 交互演示">
    <div class="demo-head">
      <div>
        <strong>Ring all-gather：每步转发一个 chunk</strong>
        <p>点按步骤或播放，观察 chunk 沿环传递；每个 rank 初始持有自己的 chunk。</p>
      </div>
      <div class="controls" aria-label="动画控制">
        <button type="button" @click="step(-1)" aria-label="上一步">◀</button>
        <button type="button" @click="toggle" :aria-label="playing ? '暂停' : '播放'">{{ playing ? '⏸' : '▶' }}</button>
        <button type="button" @click="step(1)" aria-label="下一步">▶</button>
      </div>
    </div>

    <div class="ring-viewport">
      <div class="ring-canvas">
        <svg class="ring-arrows" viewBox="0 0 520 320" aria-hidden="true">
          <defs><marker id="ring-arrow-head" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" class="arrow-head" /></marker></defs>
          <path v-for="edge in edges" :key="edge.from" :d="edge.path" class="ring-edge" :class="{ active: isActive(edge.from) }" marker-end="url(#ring-arrow-head)" />
          <text v-for="edge in edges" :key="'label-' + edge.from" :x="edge.lx" :y="edge.ly" class="edge-label" :class="{ active: isActive(edge.from) }">{{ isActive(edge.from) ? `chunk ${sentChunk(edge.from)}` : '→' }}</text>
        </svg>
        <article v-for="rank in ranks" :key="rank" class="rank-card" :class="cardClass(rank)" :style="position(rank)">
          <strong>rank {{ rank }}</strong>
          <span class="rank-caption">{{ stepIndex === 0 ? '初始' : `完成 ${stepIndex} / 3 步` }}</span>
          <div class="chunks" :aria-label="`rank ${rank} 持有的 chunk`">
            <span v-for="chunk in heldChunks(rank)" :key="chunk" class="chunk" :class="{ fresh: stepIndex > 0 && chunk === receivedChunk(rank) }">C{{ chunk }}</span>
          </div>
        </article>
      </div>
    </div>

    <div class="step-readout" aria-live="polite">
      <strong>{{ stepTitle }}</strong>
      <span>{{ stepDescription }}</span>
    </div>
    <div class="ring-footer">
      <span><i class="legend-dot"></i>本步正在发送的 chunk</span>
      <span>每步发送量：S/4；总计：3S/4 = S(N−1)/N（每 rank）</span>
    </div>
    <p class="ring-note">all-reduce 的 ring 实现由 reduce-scatter 与 all-gather 两段组成；此动画聚焦第二段，展示为什么 all-gather 只需 N−1 步。</p>
  </section>
</template>

<script setup>
import { computed, onUnmounted, ref } from 'vue'

const ranks = [0, 1, 2, 3]
const positions = [
  { left: '34px', top: '18px' },
  { left: '326px', top: '18px' },
  { left: '326px', top: '184px' },
  { left: '34px', top: '184px' },
]
const edges = [
  { from: 0, path: 'M210 67 C250 35 280 35 320 67', lx: 264, ly: 40 },
  { from: 1, path: 'M405 112 C435 132 435 150 405 177', lx: 440, ly: 148 },
  { from: 2, path: 'M320 230 C280 262 250 262 210 230', lx: 264, ly: 275 },
  { from: 3, path: 'M115 177 C85 150 85 132 115 112', lx: 62, ly: 148 },
]
const stepIndex = ref(0)
const playing = ref(false)
let timer
const stepTitle = computed(() => stepIndex.value === 0 ? '准备：每个 rank 只有自己的 chunk' : `第 ${stepIndex.value} 步：每个 rank 同时发送、接收一个 chunk`)
const stepDescription = computed(() => {
  if (stepIndex.value === 0) return '例如 rank 0 持有 C0。环上每条箭头都表示“发给右邻”，但尚未开始传输。'
  if (stepIndex.value === 3) return '三步后，每个 rank 都收齐 C0–C3。自己的 chunk 无需传递，因此是 N−1 步，而不是 N 步。'
  return `每个 rank 把上一步收到的 chunk 转发给右邻；此刻 rank 0 转发 C${sentChunk(0)}，并从左邻收到 C${receivedChunk(0)}。`
})

function heldChunks(rank) {
  return Array.from({ length: stepIndex.value + 1 }, (_, i) => (rank - i + 4) % 4).sort((a, b) => a - b)
}
function sentChunk(rank) { return (rank - stepIndex.value + 1 + 4) % 4 }
function receivedChunk(rank) { return (rank - stepIndex.value + 4) % 4 }
function isActive() { return stepIndex.value > 0 }
function cardClass(rank) { return { sending: isActive(rank), receiving: stepIndex.value > 0 && rank === 1 } }
function position(rank) { return positions[rank] }
function step(direction) { stepIndex.value = (stepIndex.value + direction + 4) % 4; pause() }
function toggle() {
  if (playing.value) return pause()
  playing.value = true
  timer = setInterval(() => { stepIndex.value = (stepIndex.value + 1) % 4 }, 1800)
}
function pause() { playing.value = false; clearInterval(timer) }
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.ring-demo { --surface:#fcfcfb; --card:#fff; --ink:#1f2328; --muted:#57606a; --line:#d0d7de; --soft:#f5f5f4; --accent:#2878c8; margin:1rem 0 1.4rem; padding:14px 16px; border:1px solid var(--line); border-radius:10px; background:var(--surface); color:var(--ink); }
.dark .ring-demo { --surface:#1a1a19; --card:#242423; --ink:#e6e6e3; --muted:#9d9d97; --line:#3d3d3b; --soft:#2b2b2a; --accent:#68a9ed; }
.demo-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
.demo-head strong { font-size:14px; }
.demo-head p,.ring-note { margin:3px 0 0; color:var(--muted); font-size:12px; line-height:1.55; }
.controls { display:flex; gap:5px; flex:none; }
.controls button { min-width:32px; border:1px solid var(--line); border-radius:6px; padding:4px 8px; color:var(--ink); background:var(--card); cursor:pointer; }
.ring-viewport { overflow-x:auto; }
.ring-canvas { position:relative; width:520px; height:320px; margin:4px auto; }
.ring-arrows { position:absolute; inset:0; width:100%; height:100%; overflow:visible; }
.ring-edge { fill:none; stroke:var(--line); stroke-width:3; transition:stroke .25s, opacity .25s; }
.ring-edge.active { stroke:var(--accent); stroke-dasharray:8 6; animation:flow .65s linear infinite; }
.arrow-head { fill:var(--accent); }
.edge-label { fill:var(--muted); font-size:11px; text-anchor:middle; paint-order:stroke; stroke:var(--surface); stroke-width:4px; stroke-linejoin:round; }
.edge-label.active { fill:var(--accent); font-weight:700; }
.rank-card { position:absolute; width:176px; min-height:82px; box-sizing:border-box; border:1px solid var(--line); border-radius:9px; padding:9px 11px; background:var(--card); transition:border-color .2s, box-shadow .2s; }
.rank-card.sending { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }
.rank-card strong { font-size:13px; }
.rank-caption { float:right; color:var(--muted); font-size:11px; }
.chunks { display:flex; gap:5px; margin-top:9px; min-height:24px; }
.chunk { display:inline-flex; align-items:center; justify-content:center; min-width:27px; height:23px; box-sizing:border-box; border:1px solid var(--line); border-radius:5px; color:var(--muted); font-size:11px; }
.chunk.fresh { border-color:var(--accent); background:var(--accent); color:white; }
.step-readout { display:grid; gap:2px; padding:9px 12px; border-radius:7px; background:var(--soft); font-size:12px; line-height:1.55; }
.step-readout strong { font-size:13px; }
.ring-footer { display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap; margin-top:9px; color:var(--muted); font-size:11px; }
.legend-dot { display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:3px; background:var(--accent); vertical-align:-1px; }
.ring-note { margin-top:6px; }
@keyframes flow { to { stroke-dashoffset:-14; } }
@media (prefers-reduced-motion:reduce) { .ring-edge.active { animation:none; } }
@media (max-width:600px) { .ring-canvas { margin-left:0; margin-right:0; } }
</style>
