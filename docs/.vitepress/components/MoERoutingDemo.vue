<template>
  <section class="moe-demo" aria-label="MoE 路由与专家并行演示">
    <strong>4 个 token 如何走完一层 MoE</strong>
    <p class="hint">2 个 rank，各持 2 个专家（rank 0：E0、E1；rank 1：E2、E3）；Top-2 路由把每个 token 送进 2 个专家。逐步观察路由、dispatch、计算与 combine。</p>
    <div class="cf-picker" aria-label="选择容量因子">
      <span>容量因子 CF：</span>
      <button v-for="opt in cfOptions" :key="opt.cf" type="button" :aria-pressed="cf === opt.cf" @click="selectCf(opt.cf)">
        {{ opt.cf }}（C={{ capacity }} 中 C={{ opt.cap }}）
      </button>
    </div>
    <div class="controls" aria-label="播放控制">
      <button type="button" @click="reset">重置</button>
      <button type="button" :disabled="phase === 0" @click="move(-1)">上一步</button>
      <button type="button" @click="toggle">{{ playing ? '暂停' : '播放' }}</button>
      <button type="button" :disabled="phase === phases.length - 1" @click="move(1)">下一步</button>
      <span aria-live="polite">{{ phase === 0 ? '准备开始' : `阶段 ${phase} / ${phases.length - 1}：${phases[phase]}` }}</span>
    </div>

    <div v-if="phase >= 1" class="block">
      <h4>① 路由（每个源 rank 本地计算）</h4>
      <table class="route-table">
        <thead><tr><th scope="col">token（源）</th><th scope="col">Top-1</th><th scope="col">Top-2</th></tr></thead>
        <tbody>
          <tr v-for="t in tokens" :key="t.id">
            <th scope="row">T{{ t.id }}（rank {{ t.src }}）</th>
            <td v-for="sel in t.routes" :key="sel.expert">E{{ sel.expert }} <small>g={{ sel.gate.toFixed(1) }}</small></td>
          </tr>
        </tbody>
      </table>
      <p class="hint">Top-2 把每个 token 复制成 2 份专家任务，共 {{ taskCount }} 份；gate 是组合权重，combine 时使用。</p>
    </div>

    <div v-if="phase >= 2" class="block">
      <h4>② dispatch：按目标 rank 打包，一次 all-to-all</h4>
      <div class="a2a">
        <div v-for="rank in rankCount" :key="rank" class="pkg">
          <strong>rank {{ rank - 1 }} 发送</strong>
          <span v-for="job in dispatchPackages[rank - 1]" :key="job.label" class="chip" :class="{ cross: job.cross }">{{ job.label }}</span>
          <em v-if="!dispatchPackages[rank - 1].some(j => j.cross)">无跨卡任务</em>
        </div>
      </div>
      <p class="hint">带边框的是跨 rank 任务：rank 0 发 T1→E2、T2→E3，rank 1 发 T3→E0、T3→E1。两个方向发送的内容互不相同——这是 A2A 与 all-gather 的区别。</p>
    </div>

    <div v-if="phase >= 3" class="block">
      <h4>③ 专家计算：容量 C={{ capacity }} 决定处理、丢弃与填充</h4>
      <div class="experts">
        <div v-for="e in experts" :key="e.id" class="expert" :class="{ hot: e.arrived.length === maxLoad && maxLoad > 0 }">
          <strong>E{{ e.id }}</strong><span>rank {{ e.rank }}</span>
          <div class="slots">
            <span v-for="(job, i) in capacitySlots(e)" :key="i" class="slot" :class="{ filled: job && !job.dropped, dropped: job && job.dropped, pad: !job }">
              {{ job ? `T${job.token}` : 'pad' }}
            </span>
          </div>
          <small>{{ e.arrived.length }} 任务 / C={{ capacity }}<template v-if="droppedByExpert(e.id)">，丢弃 T{{ droppedByExpert(e.id) }}</template></small>
        </div>
      </div>
      <p class="hint">E0 收到 {{ expertById(0).arrived.length }} 个任务而 E1 只收到 {{ expertById(1).arrived.length }} 个：整层的耗时由最忙的专家决定。超出容量的任务被丢弃（红叉），不足的槽位用 padding 占位（灰格）。</p>
    </div>

    <div v-if="phase >= 4" class="block">
      <h4>④ combine：送回源 rank，按 gate 加权求和</h4>
      <table class="route-table">
        <thead><tr><th scope="col">token</th><th scope="col">输出构成</th></tr></thead>
        <tbody>
          <tr v-for="line in combineLines" :key="line.token">
            <th scope="row">T{{ line.token }}</th>
            <td v-html="line.expr"></td>
          </tr>
        </tbody>
      </table>
      <p class="hint">被丢弃的分支不再贡献输出；是否对剩余专家重新归一化因实现而异，此处仅把丢弃分支计为 0。</p>
    </div>

    <p class="summary" aria-live="polite">{{ summary }}</p>
    <p class="note">示意假设：路由结果固定（手工设定以制造热点），gate 已算好；容量在专家入口生效，实际框架可能在 dispatch 前后以不同策略处理溢出；训练时 dispatch/combine 的反向与梯度传播未展示；省略了共享专家与重归一化细节。C = ⌈CF·Tk/N⌉，此处 T=4、k=2、N=4。</p>
  </section>
</template>

<script setup>
import { computed, onUnmounted, ref } from 'vue'

const rankCount = 2
const phases = ['准备', '路由', 'dispatch', '专家计算', 'combine']
// Hand-set Top-2 routing (gates sum to 1 per token); E0 is deliberately overloaded.
const tokens = [
  { id: 1, src: 0, routes: [{ expert: 0, gate: 0.6 }, { expert: 2, gate: 0.4 }] },
  { id: 2, src: 0, routes: [{ expert: 0, gate: 0.7 }, { expert: 3, gate: 0.3 }] },
  { id: 3, src: 1, routes: [{ expert: 0, gate: 0.5 }, { expert: 1, gate: 0.5 }] },
  { id: 4, src: 1, routes: [{ expert: 2, gate: 0.8 }, { expert: 3, gate: 0.2 }] },
]
const experts = Array.from({ length: 4 }, (_, id) => ({
  id,
  rank: id < 2 ? 0 : 1,
  arrived: tokens.filter(t => t.routes.some(r => r.expert === id)).map(t => t.id),
}))
const taskCount = tokens.length * 2
const cfOptions = [
  { cf: 1.0, cap: 2 },
  { cf: 1.5, cap: 3 },
  { cf: 2.0, cap: 4 },
]
const cf = ref(1.0)
const capacity = computed(() => cfOptions.find(o => o.cf === cf.value).cap)
const phase = ref(0)
const playing = ref(false)
let timer

// Arrival order = ascending token id; tasks beyond capacity are dropped.
const processed = computed(() => experts.map(e => ({
  ...e,
  tasks: e.arrived.map((token, i) => ({ token, dropped: i >= capacity.value })),
})))
function expertById(id) { return processed.value[id] }
function droppedByExpert(id) {
  const dropped = expertById(id).tasks.filter(t => t.dropped).map(t => t.token)
  return dropped.length ? dropped.join('、') : 0
}
function capacitySlots(e) {
  return Array.from({ length: capacity.value }, (_, i) => e.tasks[i])
}
const maxLoad = computed(() => Math.max(...processed.value.map(e => e.arrived.length)))
const droppedCount = computed(() => processed.value.reduce((sum, e) => sum + e.tasks.filter(t => t.dropped).length, 0))
const padCount = computed(() => processed.value.reduce((sum, e) => sum + Math.max(0, capacity.value - e.arrived.length), 0))

const dispatchPackages = computed(() => Array.from({ length: rankCount }, (_, rank) => {
  const jobs = []
  tokens.forEach(t => t.routes.forEach(r => {
    const target = r.expert < 2 ? 0 : 1
    if (target === rank) jobs.push({ label: `T${t.id}→E${r.expert}`, cross: t.src !== rank })
  }))
  return jobs
}))

const combineLines = computed(() => tokens.map(t => {
  const parts = []
  let droppedHtml = ''
  t.routes.forEach(r => {
    const dropped = expertById(r.expert).tasks.find(x => x.token === t.id)?.dropped
    const piece = `<code>${r.gate.toFixed(1)}·FFN<sub>E${r.expert}</sub></code>`
    if (dropped) droppedHtml += `<s>${piece}</s> `
    else parts.push(piece)
  })
  const expr = parts.join(' + ') + (droppedHtml ? ` ＝ ${parts.length ? parts.join(' + ') + ' （丢弃 ' + droppedHtml.trim() + '）' : droppedHtml.trim() + '（全部丢弃）'}` : '')
  return { token: t.id, expr }
}))

const summary = computed(() => {
  if (phase.value < 3) return 'CF 越小，容量越小，热点专家越容易丢弃任务；切换 CF 后重新步进观察第 ③ 步。'
  const dropPart = droppedCount.value ? `丢弃 ${droppedCount.value} 份任务` : '无丢弃'
  return `C=${capacity.value}：8 份专家任务中处理 ${taskCount.value - droppedCount.value} 份，${dropPart}，padding ${padCount.value} 格；最忙专家有 ${maxLoad.value} 个任务。`
})
function selectCf(value) { cf.value = value }
function pause() { playing.value = false; clearInterval(timer) }
function reset() { pause(); phase.value = 0 }
function move(delta) { pause(); phase.value = Math.max(0, Math.min(phases.length - 1, phase.value + delta)) }
function toggle() {
  if (playing.value) return pause()
  if (phase.value === phases.length - 1) phase.value = 0
  playing.value = true
  timer = setInterval(() => { phase.value++; if (phase.value === phases.length - 1) pause() }, 1800)
}
onUnmounted(pause)
</script>

<style scoped>
.moe-demo { margin: 1.2rem 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); }
.hint, .note, .summary { margin: 8px 0; color: var(--vp-c-text-2); font-size: 13px; line-height: 1.6; }
.summary { padding: 8px 10px; border-left: 3px solid var(--vp-c-brand-1); background: var(--vp-c-bg); color: var(--vp-c-text-1); }
.controls, .cf-picker { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; margin: 10px 0; font-size: 13px; }
button { border: 1px solid var(--vp-c-divider); border-radius: 6px; padding: 4px 10px; background: var(--vp-c-bg); color: var(--vp-c-text-1); cursor: pointer; }
button:disabled { opacity: .45; cursor: default; }
button[aria-pressed="true"] { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); font-weight: 700; }
button:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.block { margin-top: 12px; }
.block h4 { margin: 0 0 6px; font-size: 14px; }
.route-table { width: max-content; max-width: 100%; border-collapse: collapse; font-size: 13px; }
.route-table th, .route-table td { padding: 5px 10px; border: 1px solid var(--vp-c-divider); text-align: left; }
.route-table th { background: var(--vp-c-bg); }
.route-table td small { color: var(--vp-c-text-2); }
.a2a { display: flex; flex-wrap: wrap; gap: 8px; }
.pkg { display: flex; align-items: center; flex-wrap: wrap; gap: 5px; flex: 1 1 220px; padding: 9px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); font-size: 13px; }
.chip { padding: 2px 7px; border-radius: 5px; background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); font-size: 12px; }
.chip.cross { border: 1px dashed var(--vp-c-brand-1); font-weight: 700; }
.experts { display: flex; flex-wrap: wrap; gap: 8px; }
.expert { display: grid; gap: 4px; flex: 1 1 150px; padding: 9px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); font-size: 13px; }
.expert span, .expert small { color: var(--vp-c-text-2); font-size: 12px; }
.expert.hot { border-color: var(--vp-c-warning-1); }
.slots { display: flex; gap: 4px; flex-wrap: wrap; min-height: 24px; }
.slot { display: inline-flex; align-items: center; justify-content: center; min-width: 30px; height: 23px; border: 1px solid var(--vp-c-divider); border-radius: 5px; font-size: 11px; }
.slot.filled { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); }
.slot.dropped { background: var(--vp-c-danger-soft); color: var(--vp-c-danger-1); text-decoration: line-through; }
.slot.pad { color: var(--vp-c-text-3); border-style: dashed; }
.note { margin-top: 12px; }
</style>
