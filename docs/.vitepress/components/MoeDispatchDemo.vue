<template>
  <section class="moe-demo" aria-label="MoE 专家并行 dispatch/combine 交互演示">
    <strong>四个 token，怎么走向四位专家</strong>
    <p class="hint">2 个 rank，各持有 2 个专家（rank 0：E0、E1；rank 1：E2、E3）。每个 token 经 Top-2 路由产生 2 份专家任务，共 8 份任务。步进观察 dispatch（all-to-all）、专家计算和 combine（加权求和）。</p>
    <div class="controls" aria-label="播放控制">
      <button type="button" @click="reset">重置</button>
      <button type="button" :disabled="phase === 0" @click="move(-1)">上一步</button>
      <button type="button" @click="toggle">{{ playing ? '暂停' : '播放' }}</button>
      <button type="button" :disabled="phase === 4" @click="move(1)">下一步</button>
      <span aria-live="polite">{{ phaseNames[phase] }}</span>
    </div>
    <p class="legend">
      <span><i class="dot fwd"></i>dispatch 方向</span>
      <span><i class="dot back"></i>combine 方向</span>
      <span><i class="dot dropped"></i>溢出丢弃</span>
      <label><input type="checkbox" v-model="useCapacity" /> 启用容量限制（CF=1）</label>
    </p>

    <div class="tokens" aria-label="token 与路由结果">
      <div v-for="token in tokens" :key="token.id" class="token" :class="{ dimmed: phase >= 3 && isDropped(token.id) }">
        <strong>{{ token.id }}</strong>
        <span class="route">{{ token.routes.map(r => `E${r.expert}×${r.gate}`).join(' + ') }}</span>
        <small v-if="phase >= 3 && combineText(token.id)" v-html="combineText(token.id)"></small>
        <small v-else-if="phase >= 3" class="dropped-note">两份任务均溢出，输出为 0</small>
      </div>
    </div>

    <div class="dispatch-view" aria-live="polite">
      <div v-for="rank in 2" :key="rank" class="rank-box" :class="{ hot: phase === 2 && rank === 1 }">
        <strong>rank {{ rank - 1 }}</strong>
        <div v-for="e in 2" :key="e" class="expert">
          <b>E{{ (rank - 1) * 2 + e - 1 }}</b>
          <span class="slots">
            <i v-for="s in capacitySlots" :key="s" class="slot" :class="slotClass((rank - 1) * 2 + e - 1, s)"></i>
          </span>
          <em v-if="phase >= 1">{{ expertLoad((rank - 1) * 2 + e - 1) }}/{{ useCapacity ? capacitySlots : '∞' }}</em>
        </div>
      </div>
    </div>
    <p class="phase-note" aria-live="polite">{{ phaseNote }}</p>
    <p class="summary" aria-live="polite">{{ summary }}</p>
    <p class="note">路由权重为示意值；容量按 CF=1 且全局路由域计 8 份任务、每专家容量 2 格。启用容量后超出的任务按到达顺序丢弃（示意策略，真实实现可能改路由或重试）；丢弃任务在 combine 中按 0 参与求和，反向同样受影响。示例忽略共享专家、all-to-all 的计数交换与反向通信；这是机制示意而非某个具体框架的 dispatcher。</p>
  </section>
</template>

<script setup>
import { computed, onUnmounted, ref } from 'vue'

// 4 tokens x Top-2 over experts E0..E3 spread across 2 ranks.
// E0 deliberately gets 3 tasks (hot), E1 only 1, so capacity overflow appears.
const tokens = [
  { id: 't1', at: 0, routes: [{ expert: 0, gate: 0.6 }, { expert: 1, gate: 0.4 }] },
  { id: 't2', at: 0, routes: [{ expert: 0, gate: 0.7 }, { expert: 2, gate: 0.3 }] },
  { id: 't3', at: 1, routes: [{ expert: 0, gate: 0.5 }, { expert: 3, gate: 0.5 }] },
  { id: 't4', at: 1, routes: [{ expert: 2, gate: 0.6 }, { expert: 3, gate: 0.4 }] },
]
const taskList = tokens.flatMap(t => t.routes.map(r => ({ token: t.id, expert: r.expert, gate: r.gate })))
const expertRanks = [0, 0, 1, 1]
const CAP = 2
const phaseNames = ['0/4 准备', '1/4 路由完成', '2/4 dispatch（all-to-all）', '3/4 专家计算', '4/4 combine（加权求和）']

const phase = ref(0)
const useCapacity = ref(true)
const playing = ref(false)
let timer

const capacitySlots = CAP
function tasksFor(expert) { return taskList.filter(t => t.expert === expert) }
function isDropped(tokenId, expert) {
  if (!useCapacity.value) return false
  const queue = tasksFor(expert)
  const order = taskList.filter(t => t.expert === expert)
  return order.indexOf(taskList.find(t => t.token === tokenId && t.expert === expert)) >= CAP
}
function droppedAt(expert) {
  if (!useCapacity.value) return []
  return tasksFor(expert).slice(CAP).map(t => t.token)
}
function admitted(expert) { return tasksFor(expert).slice(0, useCapacity.value ? CAP : undefined) }
function expertLoad(expert) { return admitted(expert).length }
function slotClass(expert, slotIdx) {
  const list = admitted(expert)
  if (slotIdx <= list.length) return { filled: true, hot: expert === 0 }
  if (useCapacity.value && tasksFor(expert).length > 0) return { padding: true }
  return {}
}
function combineText(tokenId) {
  const parts = tokens.find(t => t.id === tokenId).routes
    .filter(r => !isDropped(tokenId, r.expert))
    .map(r => `${r.gate}·E${r.expert}(x)`)
  return parts.length ? `y = ${parts.join(' + ')}` : ''
}

const phaseNote = computed(() => [
  '每个 token 停留在自己的源 rank；路由表已标出各自选中的两位专家与 gate 权重。',
  '路由是本地计算：不发生数据移动。注意 E0 被 3 个任务选中，而 E1 只有 1 个——倾斜已经出现。',
  '每份专家任务被发送到专家所在 rank：t1 的一份去 rank 0 的 E0，t3 的一份跨 rank 去 rank 1 的 E3……源 rank 与专家 rank 不同正是 all-to-all 存在的理由。',
  '专家并行处理到达的任务。启用容量时 E0 只处理前 2 份，第 3 份被丢弃；关闭容量则全部处理，但热点专家耗时最长。',
  '结果沿原路送回源 rank，按 gate 权重加权求和，恢复成每个 token 的层输出。',
][phase.value])
const summary = computed(() => {
  if (phase.value === 1) {
    const loads = [0, 1, 2, 3].map(e => `E${e}:${tasksFor(e).length}`).join(' ')
    return `任务分布 ${loads}——8 份任务、4 位专家，理想均摊是每专家 2 份；实际最大 3、最小 1。`
  }
  if (phase.value === 2) return `跨 rank 任务：t2→E2、t3→E0 与 t3→E3 中目标在另一 rank 的部分；本 rank 任务无需通信。`
  if (phase.value === 3) {
    const dropped = droppedAt(0)
    return useCapacity.value && dropped.length
      ? `E0 容量 2/2，t${dropped[0]} 的任务被丢弃；E2、E3 满负荷，E1 仅 1 份任务。最忙与最闲专家的差距直接决定该层耗时。`
      : '无容量限制：所有任务都被处理，但 E0 要做 3 份计算，整层等待 E0。'
  }
  if (phase.value === 4) {
    return useCapacity.value
      ? 't3 的输出只剩 E3 一项（E0 任务被丢弃），其他 token 正常加权求和。丢弃改变了该 token 的实际计算路径。'
      : '所有 token 均按完整 Top-2 加权求和；代价是负载倾斜拖慢了整层。'
  }
  return '共 8 份专家任务待处理；先勾选/取消容量限制，再逐步观察两种模式下的差异。'
})
function pause() { playing.value = false; clearInterval(timer) }
function reset() { pause(); phase.value = 0 }
function move(delta) { pause(); phase.value = Math.max(0, Math.min(4, phase.value + delta)) }
function toggle() {
  if (playing.value) return pause()
  if (phase.value === 4) phase.value = 0
  playing.value = true
  timer = setInterval(() => { phase.value++; if (phase.value === 4) pause() }, 2000)
}
onUnmounted(pause)
</script>

<style scoped>
.moe-demo { margin: 1.2rem 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); }
.hint, .note, .phase-note, .summary { margin: 8px 0; color: var(--vp-c-text-2); font-size: 13px; line-height: 1.6; }
.controls { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; margin: 10px 0; font-size: 13px; }
button { border: 1px solid var(--vp-c-divider); border-radius: 6px; padding: 4px 10px; background: var(--vp-c-bg); cursor: pointer; }
button:disabled { opacity: .45; cursor: default; }
button:focus-visible, .legend label:focus-within { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.legend { display: flex; align-items: center; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--vp-c-text-2); }
.legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.dot.fwd { background: var(--vp-c-brand-1); }
.dot.back { background: var(--vp-c-warning-1, #b58a2e); }
.dot.dropped { background: var(--vp-c-danger-1, #cf4a4a); }
.tokens { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
.token { display: grid; gap: 3px; min-width: 168px; flex: 1 1 170px; padding: 9px 11px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); font-size: 13px; }
.token.dimmed { opacity: .62; }
.token .route { color: var(--vp-c-text-2); font-size: 12px; }
.token small { font-size: 12px; color: var(--vp-c-brand-1); }
.token small.dropped-note { color: var(--vp-c-danger-1, #cf4a4a); }
.dispatch-view { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
.rank-box { flex: 1 1 220px; display: grid; gap: 7px; padding: 10px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); }
.rank-box.hot { border-color: var(--vp-c-warning-1, #b58a2e); }
.expert { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.expert b { min-width: 28px; }
.expert em { margin-left: auto; color: var(--vp-c-text-2); font-size: 12px; font-style: normal; }
.slots { display: flex; gap: 4px; }
.slot { width: 16px; height: 16px; border: 1px solid var(--vp-c-divider); border-radius: 4px; background: var(--vp-c-bg); }
.slot.filled { background: var(--vp-c-brand-soft); border-color: var(--vp-c-brand-1); }
.slot.filled.hot { background: var(--vp-c-warning-soft); border-color: var(--vp-c-warning-1, #b58a2e); }
.slot.padding { background: transparent; border-style: dashed; }
.phase-note { padding: 8px 10px; border-left: 3px solid var(--vp-c-brand-1); background: var(--vp-c-bg); color: var(--vp-c-text-1); }
</style>
