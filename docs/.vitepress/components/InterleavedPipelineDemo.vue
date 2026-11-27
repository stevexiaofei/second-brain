<template>
  <section class="interleave-demo" aria-label="交错流水线并行演示">
    <strong>两张卡，四个虚拟 stage</strong>
    <p class="hint">模型层的顺序是 S0 → S1 → S2 → S3；每张卡持有两个不连续的 stage。选一个 microbatch，跟踪它的路径；下方调度表展示同一份调度的完整时隙。</p>
    <div class="controls" aria-label="播放控制">
      <button type="button" @click="reset">重置</button>
      <button type="button" :disabled="step === 0" @click="move(-1)">上一步</button>
      <button type="button" @click="toggle">{{ playing ? '暂停' : '播放' }}</button>
      <button type="button" :disabled="step === slots.length" @click="move(1)">下一步</button>
      <span aria-live="polite">{{ step === 0 ? '准备开始' : `已完成时隙 ${step} / ${slots.length}` }}</span>
    </div>
    <div class="batch-picker" aria-label="选择要跟踪的 microbatch">
      <span>跟踪：</span>
      <button v-for="batch in batchCount" :key="batch" type="button" :aria-pressed="selected === batch" @click="selectBatch(batch)">M{{ batch }}</button>
    </div>
    <div class="path-scroll" tabindex="0" aria-label="模型 stage 路径，可水平滚动">
      <div class="path">
        <template v-for="stage in stageCount" :key="stage">
          <div class="stage" :class="{ active: isActive(stage - 1), done: stageState(stage - 1) !== '未到达' }">
            <strong>S{{ stage - 1 }}</strong><span>rank {{ (stage - 1) % rankCount }}</span><small>{{ stageState(stage - 1) }}</small>
          </div>
          <span v-if="stage < stageCount" class="arrow" aria-hidden="true">→</span>
        </template>
      </div>
    </div>
    <p class="hint">选中的 M{{ selected }}：前向沿 S0 → S3，反向沿 S3 → S0；卡间边界也包括 S1 → S2（从 rank 1 回到 rank 0）。</p>
    <div class="ranks" aria-live="polite">
      <div v-for="rank in rankCount" :key="rank" class="rank">
        <strong>rank {{ rank - 1 }}</strong><span>持有 S{{ rank - 1 }}、S{{ rank - 1 + rankCount }}</span>
        <b>{{ actionText(rank - 1) }}</b>
      </div>
    </div>
    <p class="explain">{{ explanation }}</p>
    <div class="table-scroll" tabindex="0" aria-label="交错调度表，可水平滚动">
      <table>
        <thead><tr><th scope="col">rank / 时隙</th><th v-for="n in slots.length" :key="n" scope="col">{{ n }}</th></tr></thead>
        <tbody>
          <tr v-for="rank in rankCount" :key="rank">
            <th scope="row">rank {{ rank - 1 }}<small>S{{ rank - 1 }}·S{{ rank - 1 + rankCount }}</small></th>
            <td v-for="n in slots.length" :key="n" :class="cellClass(rank - 1, n)" :title="cellTitle(rank - 1, n)">{{ cellText(rank - 1, n) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="legend">F = 前向 · B = 反向 · — = 空闲 · 描边 = 当前时隙 · 高亮 = 选中的 M{{ selected }} 经过该 stage</p>
    <p class="note">F 表示前向，B 表示反向；M 是 microbatch，S 是虚拟 stage。每个时隙每张卡最多执行一个计算，两个 rank 可并行；依赖只能使用上一时隙之前完成的结果。示例把前向和反向均设为一个时隙，忽略通信、不均衡、重计算和优化器；这是合法的示意调度，并非所有框架的唯一实现。</p>
  </section>
</template>

<script setup>
import { computed, onUnmounted, ref } from 'vue'

const rankCount = 2
const stageCount = 4
const batchCount = 4

// Forward and backward dependencies are checked against completed slots only.
// A rank prioritizes ready backwards, then the later virtual stage.
function makeSlots() {
  const f = Array(stageCount).fill(0)
  const b = Array(stageCount).fill(0)
  const result = []
  while (b[0] < batchCount) {
    const actions = Array.from({ length: rankCount }, (_, rank) => {
      const ready = []
      for (let s = rank; s < stageCount; s += rankCount) {
        const nextB = b[s] + 1
        if (nextB <= f[s] && (s === stageCount - 1 || b[s + 1] >= nextB)) ready.push({ kind: 'B', stage: s, batch: nextB })
        const nextF = f[s] + 1
        if (nextF <= batchCount && (s === 0 || f[s - 1] >= nextF)) ready.push({ kind: 'F', stage: s, batch: nextF })
      }
      ready.sort((a, z) => (a.kind === 'B' ? 0 : 1) - (z.kind === 'B' ? 0 : 1) || z.stage - a.stage)
      return ready[0] || null
    })
    if (actions.every(action => !action)) throw new Error('Interleaved schedule stalled')
    result.push(actions)
    actions.forEach(action => {
      if (action) (action.kind === 'F' ? f : b)[action.stage]++
    })
  }
  return result
}
const slots = makeSlots()
const step = ref(0)
const selected = ref(1)
const playing = ref(false)
let timer

function stageState(stage) {
  let state = '未到达'
  for (const slot of slots.slice(0, step.value)) {
    const action = slot.find(a => a && a.stage === stage && a.batch === selected.value)
    if (action) state = action.kind === 'F' ? '前向完成' : '反向完成'
  }
  return state
}
function isActive(stage) {
  return step.value > 0 && slots[step.value - 1].some(a => a && a.stage === stage && a.batch === selected.value)
}
function actionText(rank) {
  if (!step.value) return '等待开始'
  const action = slots[step.value - 1][rank]
  return action ? `${action.kind === 'F' ? '前向' : '反向'} M${action.batch} · S${action.stage}` : '空闲 / 等待依赖'
}
function cellAction(rank, n) {
  return n <= step.value ? slots[n - 1][rank] : null
}
function cellText(rank, n) {
  const action = cellAction(rank, n)
  return action ? `${action.kind}${action.batch}·S${action.stage}` : n <= step.value ? '—' : '·'
}
function cellClass(rank, n) {
  const action = cellAction(rank, n)
  return {
    future: n > step.value,
    forward: action?.kind === 'F',
    backward: action?.kind === 'B',
    current: n === step.value,
    tracked: action?.batch === selected.value,
  }
}
function cellTitle(rank, n) {
  const action = cellAction(rank, n)
  if (!action) return n <= step.value ? `时隙 ${n}：空闲` : `时隙 ${n}`
  return `时隙 ${n}：${action.kind === 'F' ? '前向' : '反向'} M${action.batch} 于 S${action.stage}`
}
const explanation = computed(() => {
  if (!step.value) return '先看 stage 归属：S0、S2 在 rank 0；S1、S3 在 rank 1。两张卡不是各自只负责连续的一半模型。'
  if (step.value === slots.length) return '所有 microbatch 已在四个 stage 完成反向；这是一次同步流水线的排空终态。'
  const actions = slots[step.value - 1].filter(Boolean)
  if (actions.some(a => a.kind === 'B') && actions.some(a => a.kind === 'F')) return '此时两张卡同时执行不同方向的计算：一个 microbatch 正在前向，另一个正在反向。'
  if (actions.some(a => a.stage >= rankCount)) return '观察较后面的虚拟 stage：一张卡完成前面的 chunk 后，还会在之后执行自己的另一个 chunk。'
  return '当前时隙的计算完成后，其结果才能供下一时隙中有依赖的 stage 使用。'
})
function pause() { playing.value = false; clearInterval(timer) }
function reset() { pause(); step.value = 0 }
function move(delta) { pause(); step.value = Math.max(0, Math.min(slots.length, step.value + delta)) }
function selectBatch(batch) { pause(); selected.value = batch }
function toggle() {
  if (playing.value) return pause()
  if (step.value === slots.length) step.value = 0
  playing.value = true
  timer = setInterval(() => { step.value++; if (step.value === slots.length) pause() }, 1500)
}
onUnmounted(pause)
</script>

<style scoped>
.interleave-demo { margin: 1.2rem 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); }
.hint, .note, .explain { margin: 8px 0; color: var(--vp-c-text-2); font-size: 13px; line-height: 1.6; }
.controls, .batch-picker { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; margin: 10px 0; font-size: 13px; }
button { border: 1px solid var(--vp-c-divider); border-radius: 6px; padding: 4px 10px; background: var(--vp-c-bg); cursor: pointer; }
button:disabled { opacity: .45; cursor: default; }
button[aria-pressed="true"] { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); font-weight: 700; }
button:focus-visible, .path-scroll:focus-visible, .table-scroll:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.path-scroll, .table-scroll { max-width: 100%; overflow-x: auto; }
.table-scroll table { width: max-content; border-collapse: collapse; margin: 10px 0 4px; font-size: 12px; line-height: 1.3; }
.table-scroll th, .table-scroll td { padding: 5px 6px; border: 1px solid var(--vp-c-divider); text-align: center; white-space: nowrap; }
.table-scroll th { background: var(--vp-c-bg); }
.table-scroll th small { display: block; color: var(--vp-c-text-2); font-weight: 400; font-size: 11px; }
.table-scroll td { min-width: 40px; background: var(--vp-c-bg); }
.table-scroll td.future { color: var(--vp-c-text-3); }
.table-scroll td.forward { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); }
.table-scroll td.backward { background: var(--vp-c-warning-soft); }
.table-scroll td.tracked { font-weight: 700; text-decoration: underline 1px; text-underline-offset: 3px; }
.table-scroll td.current { outline: 2px solid var(--vp-c-brand-1); outline-offset: -2px; }
.legend { margin: 4px 0 8px; color: var(--vp-c-text-2); font-size: 12px; }
.path { display: flex; align-items: center; width: max-content; gap: 7px; padding: 6px 2px; }
.stage { display: grid; gap: 2px; min-width: 98px; padding: 9px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); font-size: 13px; }
.stage span, .stage small { color: var(--vp-c-text-2); font-size: 12px; }
.stage.done { background: var(--vp-c-brand-soft); }
.stage.active { outline: 2px solid var(--vp-c-brand-1); }
.arrow { color: var(--vp-c-text-2); }
.ranks { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.rank { display: grid; gap: 3px; flex: 1 1 180px; padding: 10px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); font-size: 13px; }
.rank span { color: var(--vp-c-text-2); font-size: 12px; }
.explain { padding: 8px 10px; border-left: 3px solid var(--vp-c-brand-1); background: var(--vp-c-bg); color: var(--vp-c-text-1); }
</style>
