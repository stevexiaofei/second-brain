<template>
  <div class="tp-fsdp-demo">
    <div class="demo-top">
      <div class="phase-tabs" role="tablist" aria-label="训练阶段">
        <button
          v-for="(p, i) in phases"
          :key="i"
          role="tab"
          :aria-selected="i === phase"
          class="phase-tab"
          :class="{ active: i === phase }"
          @click="go(i)"
        >
          {{ p.tab }}
        </button>
      </div>
      <div class="play-ctrl">
        <button class="ctrl" aria-label="上一步" title="上一步" @click="step(-1)">◀</button>
        <button class="ctrl" :aria-label="playing ? '暂停' : '播放'" :title="playing ? '暂停' : '播放'" @click="togglePlay">
          {{ playing ? '⏸' : '▶' }}
        </button>
        <button class="ctrl" aria-label="下一步" title="下一步" @click="step(1)">▶</button>
      </div>
    </div>

    <div class="mesh-viewport">
      <div class="mesh-wrap">
      <svg class="arrows" width="640" height="470" viewBox="0 0 640 470" aria-hidden="true">
        <defs>
          <marker id="ah-tp" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto-start-reverse">
            <path d="M1,1 L8,4.5 L1,8 Z" class="mk-tp" />
          </marker>
          <marker id="ah-dp" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto-start-reverse">
            <path d="M1,1 L8,4.5 L1,8 Z" class="mk-dp" />
          </marker>
        </defs>
        <line
          v-for="(l, i) in tpLines"
          :key="'tp' + i"
          :x1="l.x1" :y1="l.y1" :x2="l.x2" :y2="l.y2"
          class="arr tp"
          :class="{ flow: flow === 'tp' }"
          marker-start="url(#ah-tp)"
          marker-end="url(#ah-tp)"
        />
        <line
          v-for="(l, i) in dpLines"
          :key="'dp' + i"
          :x1="l.x1" :y1="l.y1" :x2="l.x2" :y2="l.y2"
          class="arr dp"
          :class="{ flow: flow === 'dp' }"
          marker-start="url(#ah-dp)"
          marker-end="url(#ah-dp)"
        />
        <text x="320" y="126" text-anchor="middle" class="line-label tp-label">TP：同一批数据，切分协作</text>
        <text x="320" y="356" text-anchor="middle" class="line-label tp-label">TP：同一批数据，切分协作</text>
        <text x="130" y="254" text-anchor="middle" class="line-label dp-label" transform="rotate(-90 130 254)">DP/FSDP 组</text>
        <text x="510" y="254" text-anchor="middle" class="line-label dp-label" transform="rotate(-90 510 254)">DP/FSDP 组</text>
      </svg>

      <button
        v-for="c in cards"
        :key="c.rank"
        class="gpu-card"
        :class="{
          selected: selected === c.rank,
          'peer-tp': selected !== null && isTpPeer(c),
          'peer-dp': selected !== null && isDpPeer(c),
        }"
        :style="{ left: c.x + 'px', top: c.y + 'px' }"
        :aria-label="`GPU ${c.rank}，dp${c.dp} tp${c.tp}`"
        @click="select(c.rank)"
      >
        <span class="gpu-head">GPU {{ c.rank }}<em>dp{{ c.dp }}·tp{{ c.tp }}</em></span>
        <span class="gpu-body">{{ cardText(c) }}</span>
        <span class="gpu-data">数据 {{ c.data }}</span>
      </button>
      </div>
    </div>

    <div class="phase-desc">
      <strong>{{ phases[phase].title }}</strong>
      <span v-html="phases[phase].desc"></span>
    </div>

    <div v-if="selected !== null" class="peer-info">
      {{ peerInfo(selected) }}
      <button class="ctrl small" aria-label="取消选择" @click="selected = null">✕</button>
    </div>

    <div class="legend">
      <span><i class="sw tp"></i>TP 轴：层内协作，每层都发生</span>
      <span><i class="sw dp"></i>DP/FSDP 轴：副本间同步，恢复参数 / 归约梯度</span>
    </div>
    <p class="demo-note">
      <code>optimizer.step()</code> 更新各 rank 常驻分片（图中未画）；下一个 step 从阶段①重新开始。
      点击任一 GPU 查看它所在的通信组。
    </p>
  </div>
</template>

<script setup>
import { computed, onUnmounted, ref } from 'vue'

const phases = [
  {
    tab: '① 分片驻留',
    flow: null,
    title: '初始形态：FSDP 分片常驻',
    desc: '参数先沿 TP 轴切成 W₀ / W₁，每个 TP 分片再沿 DP 轴切成 a / b 两份。平时显存里只有这些分片，没有完整参数。',
  },
  {
    tab: '② 前向 all-gather',
    flow: 'dp',
    title: '前向：沿 DP 轴 all-gather 参数',
    desc: 'rank0 ↔ rank2 互换分片，各自临时恢复完整 W₀；rank1 ↔ rank3 恢复 W₁。这是 <code>fully_shard</code> 的前向钩子，通信只发生在竖向。',
  },
  {
    tab: '③ TP 协作计算',
    flow: 'tp',
    title: '前向：TP 层内协作计算',
    desc: '恢复参数后，每个 TP 组合作计算同一批数据：fc1 列并行、fc2 行并行，中间激活沿横向通信。对应 <code>parallelize_module(tp_plan)</code>。',
  },
  {
    tab: '④ 反向 AR',
    flow: 'tp',
    title: '反向：TP 轴同步激活梯度',
    desc: '行/列并行的反向按逻辑梯度语义做 all-reduce，把激活输入梯度在 TP 组内累加回完整。此时参数自身的梯度还留在各 rank 本地。',
  },
  {
    tab: '⑤ reduce-scatter',
    flow: 'dp',
    title: '反向：沿 DP 轴 reduce-scatter 梯度',
    desc: '同一参数分片的不同数据副本梯度（rank0 与 rank2 的 W₀ 梯度）归约并切分，各 rank 只保留自己的梯度分片，回到常驻形态。由 <code>loss.backward()</code> 触发。',
  },
]

const cards = [
  { rank: 0, x: 40,  y: 70,  dp: 0, tp: 0, data: 'A', tpGroup: [0, 1], dpGroup: [0, 2] },
  { rank: 1, x: 420, y: 70,  dp: 0, tp: 1, data: 'A', tpGroup: [0, 1], dpGroup: [1, 3] },
  { rank: 2, x: 40,  y: 300, dp: 1, tp: 0, data: 'B', tpGroup: [2, 3], dpGroup: [0, 2] },
  { rank: 3, x: 420, y: 300, dp: 1, tp: 1, data: 'B', tpGroup: [2, 3], dpGroup: [1, 3] },
]

const tpLines = [
  { x1: 228, y1: 135, x2: 412, y2: 135 },
  { x1: 228, y1: 365, x2: 412, y2: 365 },
]
const dpLines = [
  { x1: 130, y1: 208, x2: 130, y2: 292 },
  { x1: 510, y1: 208, x2: 510, y2: 292 },
]

const phase = ref(0)
const playing = ref(false)
const selected = ref(null)
let timer = null

const flow = computed(() => phases[phase.value].flow)

function go(i) {
  phase.value = i
  pause()
}
function step(d) {
  phase.value = (phase.value + d + phases.length) % phases.length
  pause()
}
function togglePlay() {
  playing.value = !playing.value
  if (playing.value) {
    timer = setInterval(() => {
      phase.value = (phase.value + 1) % phases.length
    }, 2600)
  } else {
    clearInterval(timer)
  }
}
function pause() {
  playing.value = false
  clearInterval(timer)
}
onUnmounted(() => clearInterval(timer))

function shardLetter(c) {
  return c.dp === 0 ? 'a' : 'b'
}
function cardText(c) {
  const sh = shardLetter(c)
  switch (phase.value) {
    case 0: return `常驻 W${'₀₁'[c.tp]} 分片 ${sh}`
    case 1: return `互换分片 → 完整 W${'₀₁'[c.tp]}`
    case 2: return `fc1/fc2 与 GPU${peerOf(c)} 共算`
    case 3: return `反向：AR 激活梯度`
    case 4: return `留 W${'₀₁'[c.tp]} 梯度分片 ${sh}`
  }
}
function peerOf(c) {
  return c.tpGroup[0] === c.rank ? c.tpGroup[1] : c.tpGroup[0]
}
function isTpPeer(c) {
  return selected.value !== null && selected.value !== c.rank &&
    c.tpGroup.includes(selected.value)
}
function isDpPeer(c) {
  return selected.value !== null && selected.value !== c.rank &&
    c.dpGroup.includes(selected.value)
}
function select(rank) {
  selected.value = selected.value === rank ? null : rank
  pause()
}
function peerInfo(rank) {
  const c = cards[rank]
  return `GPU ${rank}（dp${c.dp}·tp${c.tp}，数据 ${c.data}）→ TP 组 [${c.tpGroup.join(', ')}]：不同参数分片、合作计算；DP/FSDP 组 [${c.dpGroup.join(', ')}]：同一参数分片的不同副本，all-gather / reduce-scatter 在此发生。`
}
</script>

<style scoped>
.tp-fsdp-demo {
  --d-surface: #fcfcfb;
  --d-card: #ffffff;
  --d-ink: #1f2328;
  --d-ink2: #57606a;
  --d-line: #d0d7de;
  --d-soft: #f5f5f4;
  --c-tp: #eb6834;
  --c-dp: #2a78d6;
  margin: 1.2rem 0 1.4rem;
  border: 1px solid var(--d-line);
  border-radius: 10px;
  background: var(--d-surface);
  padding: 14px 16px 12px;
  max-width: 720px;
}
.dark .tp-fsdp-demo {
  --d-surface: #1a1a19;
  --d-card: #242423;
  --d-ink: #e6e6e3;
  --d-ink2: #9d9d97;
  --d-line: #3d3d3b;
  --d-soft: #2b2b2a;
  --c-tp: #d95926;
  --c-dp: #3987e5;
}

.demo-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}
.phase-tabs {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.phase-tab {
  font-size: 12.5px;
  padding: 4px 9px;
  border-radius: 14px;
  border: 1px solid var(--d-line);
  background: var(--d-card);
  color: var(--d-ink2);
  cursor: pointer;
}
.phase-tab.active {
  background: var(--d-soft);
  color: var(--d-ink);
  font-weight: 600;
  border-color: var(--d-ink2);
}
.play-ctrl {
  display: flex;
  gap: 4px;
}
.ctrl {
  min-width: 30px;
  padding: 3px 8px;
  border-radius: 6px;
  border: 1px solid var(--d-line);
  background: var(--d-card);
  color: var(--d-ink);
  cursor: pointer;
  font-size: 13px;
  line-height: 1.4;
}
.ctrl:hover { border-color: var(--d-ink2); }
.ctrl.small { min-width: 22px; padding: 1px 6px; margin-left: 8px; font-size: 11px; }

.mesh-viewport {
  width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: thin;
}
.mesh-wrap {
  position: relative;
  width: 640px;
  height: 470px;
  margin: 0 auto;
}
.arrows {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.arr {
  stroke-width: 2.5;
  fill: none;
  opacity: 0.18;
  transition: opacity 0.3s;
}
.arr.tp { stroke: var(--c-tp); }
.arr.dp { stroke: var(--c-dp); }
.arr.flow { opacity: 1; stroke-dasharray: 9 7; animation: dashflow 0.7s linear infinite; }
.mk-tp { fill: var(--c-tp); }
.mk-dp { fill: var(--c-dp); }
.line-label {
  font-size: 12px;
  font-weight: 600;
  paint-order: stroke;
  stroke: var(--d-surface);
  stroke-width: 4px;
  stroke-linejoin: round;
}
.tp-label { fill: var(--c-tp); }
.dp-label { fill: var(--c-dp); }
@keyframes dashflow {
  to { stroke-dashoffset: -16; }
}
@media (prefers-reduced-motion: reduce) {
  .arr.flow { animation: none; }
}

.gpu-card {
  position: absolute;
  width: 180px;
  height: 130px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1.5px solid var(--d-line);
  background: var(--d-card);
  color: var(--d-ink);
  text-align: left;
  cursor: pointer;
  transition: border-color 0.25s, box-shadow 0.25s;
}
.gpu-card:hover { border-color: var(--d-ink2); }
.gpu-card.selected {
  border-color: var(--d-ink);
  box-shadow: 0 0 0 2px var(--d-soft), 0 2px 8px rgba(0, 0, 0, 0.12);
}
.gpu-card.peer-tp { border-color: var(--c-tp); border-style: dashed; }
.gpu-card.peer-dp { border-color: var(--c-dp); border-style: dashed; }
.gpu-head {
  font-weight: 700;
  font-size: 14px;
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}
.gpu-head em {
  font-style: normal;
  font-size: 11px;
  color: var(--d-ink2);
  font-weight: 400;
}
.gpu-body {
  font-size: 12.5px;
  line-height: 1.45;
  min-height: 2.6em;
  color: var(--d-ink);
}
.gpu-data {
  font-size: 11px;
  color: var(--d-ink2);
  margin-top: auto;
}

.phase-desc {
  font-size: 13.5px;
  line-height: 1.65;
  color: var(--d-ink2);
  background: var(--d-soft);
  border-radius: 8px;
  padding: 10px 14px;
  margin-top: 4px;
}
.phase-desc strong {
  display: block;
  color: var(--d-ink);
  margin-bottom: 2px;
  font-size: 14px;
}
.peer-info {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--d-ink);
  background: var(--d-card);
  border: 1px dashed var(--d-ink2);
  border-radius: 8px;
  padding: 8px 12px;
}
.legend {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  margin-top: 10px;
  font-size: 12px;
  color: var(--d-ink2);
}
.legend .sw {
  display: inline-block;
  width: 22px;
  height: 0;
  border-top: 3px solid;
  margin-right: 6px;
  vertical-align: middle;
  border-radius: 2px;
}
.sw.tp { border-color: var(--c-tp); }
.sw.dp { border-color: var(--c-dp); }
.demo-note {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--d-ink2);
}
@media (max-width: 700px) {
  .mesh-wrap { margin-left: 0; margin-right: 0; }
}
</style>
