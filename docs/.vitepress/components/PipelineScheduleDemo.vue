<template>
  <section class="pipeline-demo" aria-label="GPipe 和 1F1B 调度对照演示">
    <strong>同一批数据，两种调度</strong>
    <p class="intro">4 个 stage、6 个 microbatch；每列是一个等长时隙。逐步观察前向、反向和每个 stage 尚未释放的激活。</p>
    <div class="controls" aria-label="播放控制">
      <button type="button" @click="reset">重置</button>
      <button type="button" :disabled="step === 0" @click="move(-1)">上一步</button>
      <button type="button" @click="toggle">{{ playing ? '暂停' : '播放' }}</button>
      <button type="button" :disabled="step === totalSteps" @click="move(1)">下一步</button>
      <span aria-live="polite">{{ step === 0 ? '准备开始' : `已完成时隙 ${step} / ${totalSteps}` }}</span>
    </div>
    <p class="legend">F = 前向并保存激活 · B = 反向并释放激活 · — = 空闲 · <b>描边</b> = 当前时隙</p>
    <div v-for="schedule in schedules" :key="schedule.name" class="schedule">
      <div class="schedule-header"><strong>{{ schedule.name }}</strong><span>{{ schedule.description }}</span></div>
      <div class="timeline-scroll" tabindex="0" :aria-label="`${schedule.name} 调度表，可水平滚动`">
        <table>
          <thead><tr><th scope="col">stage / 时隙</th><th v-for="n in totalSteps" :key="n" scope="col">{{ n }}</th><th scope="col">驻留激活</th></tr></thead>
          <tbody>
            <tr v-for="stage in stageCount" :key="stage">
              <th scope="row">stage {{ stage - 1 }}</th>
              <td v-for="n in totalSteps" :key="n" :class="cellClass(schedule, stage - 1, n)">{{ cellText(schedule, stage - 1, n) }}</td>
              <td class="resident">{{ resident(schedule, stage - 1).join('、') || '无' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="summary">{{ step === 0 ? '尚未执行，所有 stage 均无驻留激活。' : `当前累计空闲：${idleCount(schedule)} 个 stage·时隙；stage 0 驻留：${resident(schedule, 0).length} 个 microbatch。` }}</p>
    </div>
    <p class="note">驻留列表显示已完成前向、尚未完成反向的 microbatch；它是数量示意，不等于显存字节数。切换时隙后展示该时隙结束的状态。该模型忽略通信、重计算和不均衡，不代表具体实现的唯一执行顺序。</p>
  </section>
</template>

<script setup>
import { onUnmounted, ref } from 'vue'

const stageCount = 4
const microbatches = 6

// Each slot reads only results completed in earlier slots. For 1F1B,
// cap outstanding activations at p - stage and prioritize ready backwards.
function makeSchedule(isOneFOneB) {
  const forward = Array(stageCount).fill(0)
  const backward = Array(stageCount).fill(0)
  const slots = []
  while (backward[0] < microbatches) {
    const allForwardDone = forward[stageCount - 1] === microbatches
    const actions = forward.map((f, s) => {
      const nextB = backward[s] + 1
      const bReady = nextB <= f && (s === stageCount - 1 || backward[s + 1] >= nextB)
      if (bReady && (isOneFOneB || allForwardDone)) return { kind: 'B', batch: nextB }
      const nextF = f + 1
      const fReady = nextF <= microbatches && (s === 0 || forward[s - 1] >= nextF)
      if (fReady && (!isOneFOneB || f - backward[s] < stageCount - s)) return { kind: 'F', batch: nextF }
      return null
    })
    if (actions.every(action => !action)) throw new Error('Pipeline schedule stalled')
    slots.push(actions)
    actions.forEach((action, s) => {
      if (action?.kind === 'F') forward[s]++
      if (action?.kind === 'B') backward[s]++
    })
  }
  return slots
}

const schedules = [
  { name: 'GPipe', description: '全部前向完成后开始反向', slots: makeSchedule(false) },
  { name: '同步 1F1B', description: '预热后交替前向与反向', slots: makeSchedule(true) },
]
const totalSteps = schedules[0].slots.length
const step = ref(0)
const playing = ref(false)
let timer

function cellText(schedule, stage, n) {
  if (n > step.value) return '·'
  const action = schedule.slots[n - 1]?.[stage]
  return action ? `${action.kind}${action.batch}` : '—'
}
function cellClass(schedule, stage, n) {
  const action = n <= step.value && schedule.slots[n - 1]?.[stage]
  return { future: n > step.value, forward: action?.kind === 'F', backward: action?.kind === 'B', current: n === step.value }
}
function resident(schedule, stage) {
  const held = new Set()
  schedule.slots.slice(0, step.value).forEach(slot => {
    const action = slot[stage]
    if (action?.kind === 'F') held.add(action.batch)
    if (action?.kind === 'B') held.delete(action.batch)
  })
  return [...held].map(batch => `M${batch}`)
}
function idleCount(schedule) {
  return schedule.slots.slice(0, step.value).reduce((sum, slot) => sum + slot.filter(action => !action).length, 0)
}
function pause() { playing.value = false; clearInterval(timer) }
function reset() { pause(); step.value = 0 }
function move(direction) { pause(); step.value = Math.max(0, Math.min(totalSteps, step.value + direction)) }
function toggle() {
  if (playing.value) return pause()
  if (step.value === totalSteps) step.value = 0
  playing.value = true
  timer = setInterval(() => {
    step.value++
    if (step.value >= totalSteps) pause()
  }, 1200)
}
onUnmounted(pause)
</script>

<style scoped>
.pipeline-demo { margin: 1.2rem 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); }
.intro, .legend, .note, .summary { margin: 6px 0; color: var(--vp-c-text-2); font-size: 13px; line-height: 1.6; }
.controls { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; margin: 12px 0; font-size: 13px; }
.controls button { border: 1px solid var(--vp-c-divider); border-radius: 6px; padding: 4px 10px; background: var(--vp-c-bg); cursor: pointer; }
.controls button:disabled { opacity: .45; cursor: default; }
.controls button:focus-visible, .timeline-scroll:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.schedule { margin-top: 14px; }
.schedule-header { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; font-size: 14px; }
.schedule-header span { color: var(--vp-c-text-2); font-size: 12px; }
.timeline-scroll { max-width: 100%; overflow-x: auto; }
table { display: table; width: max-content; border-collapse: collapse; margin: 7px 0; font-size: 12px; line-height: 1.3; }
th, td { padding: 5px 6px; border: 1px solid var(--vp-c-divider); text-align: center; white-space: nowrap; }
th { background: var(--vp-c-bg); }
td { min-width: 27px; background: var(--vp-c-bg); }
td.future { color: var(--vp-c-text-3); }
td.forward { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); font-weight: 700; }
td.backward { background: var(--vp-c-warning-soft); color: var(--vp-c-text-1); font-weight: 700; }
td.current { outline: 2px solid var(--vp-c-brand-1); outline-offset: -2px; }
td.resident { min-width: 105px; text-align: left; }
.note { margin-top: 12px; }
</style>
