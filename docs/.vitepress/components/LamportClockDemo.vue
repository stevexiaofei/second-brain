<script setup>
import { computed, onUnmounted, ref, useId } from 'vue'
import { LAMPORT_SCENARIOS, getLamportState } from './lamportClockModel.mjs'

const mode = ref('message')
const index = ref(0)
const playing = ref(false)
const selected = ref(null)
const markerId = `lamport-arrow-${useId()}`
const scenario = computed(() => LAMPORT_SCENARIOS[mode.value])
const state = computed(() => getLamportState(mode.value, index.value))
const detail = computed(() => state.value.events.find(event => event.id === selected.value) || state.value.current)
const last = computed(() => index.value === scenario.value.steps.length - 1)
const y = process => process === 'P' ? 100 : 280
let timer
function pause() { playing.value = false; clearInterval(timer); timer = undefined }
function go(step) { pause(); selected.value = null; index.value = Math.max(0, Math.min(scenario.value.steps.length - 1, step)) }
function reset() { go(0) }
function switchMode(value) { pause(); selected.value = null; index.value = 0; mode.value = value }
function toggle() {
  if (playing.value) return pause()
  if (last.value) return
  selected.value = null
  playing.value = true
  timer = setInterval(() => { index.value++; if (last.value) pause() }, 3200)
}
function inspect(id) { pause(); selected.value = id }
function messagePath(message) { return `M ${message.source.x} ${y(message.source.process)} L ${message.target.x} ${y(message.target.process)}` }
onUnmounted(pause)
</script>

<template>
  <section class="lamport-demo" aria-label="Lamport 逻辑时钟交互演示">
    <strong>把事件编号一步步算出来</strong>
    <div class="mode-buttons" aria-label="选择对照场景">
      <button v-for="(item, key) in LAMPORT_SCENARIOS" :key="key" type="button" :aria-pressed="mode === key" @click="switchMode(key)">{{ item.title }}</button>
    </div>
    <p class="intro">{{ scenario.intro }}</p>
    <div class="controls" aria-label="播放控制">
      <button type="button" @click="reset">重置</button>
      <button type="button" :disabled="index === 0" @click="go(index - 1)">上一步</button>
      <button type="button" :disabled="last && !playing" @click="toggle">{{ playing ? '暂停' : '播放' }}</button>
      <button type="button" :disabled="last" @click="go(index + 1)">下一步</button>
    </div>
    <nav class="steps" aria-label="跳到指定阶段">
      <button v-for="(step, n) in scenario.steps" :key="n" type="button" :aria-current="index === n ? 'step' : undefined" @click="go(n)">{{ n }}. {{ step.title }}</button>
    </nav>
    <div class="clocks" aria-label="当前进程计数器">
      <div><span>P 当前计数器</span><strong>{{ state.clocks.P }}</strong></div>
      <div><span>Q 当前计数器</span><strong>{{ state.clocks.Q }}</strong></div>
    </div>
    <div class="explanation" aria-live="polite" aria-atomic="true">
      <strong>第 {{ index }} 步：{{ state.title }}</strong>
      <p>{{ state.description }}</p>
    </div>
    <div class="timeline-scroll" tabindex="0" aria-label="事件时间线，可横向滚动；事件列表在图下方">
      <svg :key="`${mode}-${index}`" viewBox="0 0 700 365" class="timeline" aria-hidden="true">
        <defs><marker :id="markerId" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>
        <text x="25" y="45" class="axis-label">向右只表示展示顺序，不是物理时间刻度</text>
        <g v-for="process in ['P', 'Q']" :key="process">
          <text x="25" :y="y(process) + 6" class="process-label">{{ process }}</text>
          <line x1="80" x2="660" :y1="y(process)" :y2="y(process)" class="lifeline" :marker-end="`url(#${markerId})`" />
        </g>
        <g v-for="message in state.messages" :key="message.id">
          <template v-if="message.target">
            <path :d="messagePath(message)" class="message-line" :marker-end="`url(#${markerId})`" />
            <text :x="(message.source.x + message.target.x) / 2 + 8" y="190" class="message-label">{{ message.id }}：携带 {{ message.source.clock }}</text>
            <circle v-if="state.current?.id === message.target.id" :key="`${mode}-${index}`" r="6" class="packet">
              <animateMotion :path="messagePath(message)" dur="1.3s" fill="freeze" />
              <animate attributeName="opacity" values="1;1;0" keyTimes="0;0.9;1" dur="1.3s" fill="freeze" />
            </circle>
          </template>
          <text v-else :x="message.source.x + 10" :y="y(message.source.process) + (message.source.process === 'P' ? 55 : -50)" class="pending">{{ message.id }} 在途，携带 {{ message.source.clock }}</text>
        </g>
        <g v-for="event in state.events" :key="event.id" :class="['event', { current: state.current?.id === event.id, inspected: selected === event.id }]">
          <circle :cx="event.x" :cy="y(event.process)" r="20" />
          <text :x="event.x" :y="y(event.process) + 6" text-anchor="middle" class="event-clock">{{ event.clock }}</text>
          <text :x="event.x" :y="y(event.process) + (event.process === 'P' ? -35 : 43)" text-anchor="middle" class="event-label">{{ event.label }}</text>
        </g>
        <text v-if="mode === 'independent'" x="95" y="190" class="no-message">没有跨进程消息箭头；不存在跨进程的因果路径</text>
        <text x="25" y="350" class="axis-label">圆圈里的数 = 该事件的 Lamport 编号；沿同一进程或消息箭头前进，编号增大。</text>
      </svg>
    </div>
    <div class="event-buttons" aria-label="选择已发生事件查看计算">
      <span v-if="!state.events.length">尚无事件。点“下一步”开始。</span>
      <button v-for="event in state.events" :key="event.id" type="button" :aria-pressed="detail?.id === event.id" @click="inspect(event.id)">{{ event.process }}：{{ event.label }}（编号 {{ event.clock }}）</button>
    </div>
    <div v-if="detail" class="calculation">
      <strong>{{ detail.process }} · {{ detail.label }}</strong>
      <p>{{ detail.calculation }}</p>
      <span>这是事件的编号，不是“现在几点”，也不是从全局序号池领取的号码。</span>
    </div>
    <p v-if="last" class="conclusion"><strong>这一轮说明：</strong>{{ scenario.conclusion }}</p>
    <p class="boundary">简化假设：只有 P、Q 两个进程，消息可靠且按图送达；每步只展示一个事件。按钮的点击先后不构成跨进程通信，动画时长也不代表真实延迟。</p>
  </section>
</template>

<style scoped>
.lamport-demo { min-width: 0; margin: 1.3rem 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 12px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); overflow-wrap: anywhere; }
.mode-buttons, .controls, .steps, .event-buttons { display: flex; flex-wrap: wrap; gap: 7px; margin: 12px 0; }
button { max-width: 100%; padding: 6px 10px; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); font-size: 13px; line-height: 1.5; text-align: left; cursor: pointer; }
button[aria-pressed="true"], button[aria-current="step"] { border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); font-weight: 700; }
button:disabled { opacity: .45; cursor: default; }
button:focus-visible, .timeline-scroll:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.intro, .boundary { font-size: 13px; color: var(--vp-c-text-2); line-height: 1.7; }
.clocks { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.clocks > div { display: flex; align-items: center; justify-content: space-between; gap: 6px; padding: 10px 14px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); }
.clocks span { font-size: 13px; }
.clocks strong { font-size: 27px; color: var(--vp-c-brand-1); }
.explanation, .calculation { margin-top: 12px; padding: 12px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); font-size: 14px; line-height: 1.7; }
.explanation p, .calculation p { margin: 6px 0 0; }
.calculation span { display: block; margin-top: 7px; font-size: 12px; color: var(--vp-c-text-2); }
.timeline-scroll { max-width: 100%; overflow-x: auto; margin-top: 12px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); }
.timeline { display: block; width: 100%; min-width: 640px; height: auto; }
.timeline marker path { fill: var(--vp-c-text-2); }
.lifeline { stroke: var(--vp-c-text-2); stroke-width: 1.5; }
.axis-label, .pending { fill: var(--vp-c-text-2); font-size: 13px; }
.process-label { fill: var(--vp-c-text-1); font-size: 22px; font-weight: 700; }
.message-line { fill: none; stroke: var(--vp-c-brand-1); stroke-width: 2.5; stroke-dasharray: 6 3; }
.message-label { fill: var(--vp-c-text-1); font-size: 13px; paint-order: stroke; stroke: var(--vp-c-bg); stroke-width: 5px; stroke-linejoin: round; }
.event circle { fill: var(--vp-c-bg-soft); stroke: var(--vp-c-text-2); stroke-width: 2; }
.event.current circle, .event.inspected circle { stroke: var(--vp-c-brand-1); stroke-width: 3; fill: var(--vp-c-brand-soft); }
.event-clock { fill: var(--vp-c-text-1); font-size: 18px; font-weight: 700; }
.event-label, .no-message { fill: var(--vp-c-text-1); font-size: 14px; }
.packet { fill: var(--vp-c-brand-1); stroke: var(--vp-c-bg); stroke-width: 2; }
.event-buttons { font-size: 13px; }
.conclusion { padding: 12px; border-left: 3px solid var(--vp-c-brand-1); background: var(--vp-c-brand-soft); font-size: 14px; line-height: 1.8; }
@media (max-width: 400px) { .clocks > div { flex-direction: column; align-items: flex-start; } }
@media (prefers-reduced-motion: reduce) { .packet { display: none; } }
</style>
