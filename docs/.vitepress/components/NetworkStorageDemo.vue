<template>
  <section class="storage-demo" :aria-label="scenario.title">
    <header>
      <strong>{{ scenario.title }}</strong>
      <p class="network-legend">网络标签：<b>public</b> 客户端可达的存储网络 · <b>cluster</b> 可选的独立后端网络 · 本地 = 不经过网络</p>
    </header>

    <div class="controls" aria-label="演示控制">
      <button type="button" @click="reset">重置</button>
      <button type="button" :disabled="stepIndex === 0" @click="goTo(stepIndex - 1)">上一步</button>
      <button type="button" :disabled="!playing && stepIndex === scenario.steps.length - 1" @click="toggle">{{ playing ? '暂停' : '播放' }}</button>
      <button type="button" :disabled="stepIndex === scenario.steps.length - 1" @click="goTo(stepIndex + 1)">下一步</button>
    </div>

    <nav class="steps" aria-label="选择演示阶段">
      <button v-for="(item, index) in scenario.steps" :key="index" type="button" :class="{ selected: index === stepIndex }" :aria-current="index === stepIndex ? 'step' : undefined" @click="goTo(index)">
        {{ index + 1 }}. {{ item.title }}
      </button>
    </nav>

    <div class="stage" aria-live="polite" aria-atomic="true">
      <div class="stage-heading"><span>阶段 {{ stepIndex + 1 }} / {{ scenario.steps.length }}</span><strong>{{ step.title }}</strong></div>
      <p>{{ step.description }}</p>
      <div class="nodes" aria-label="角色状态">
        <div v-for="node in scenario.nodes" :key="node.id" class="node" :class="{ active: step.activeNodes.includes(node.id) }">
          <strong>{{ node.label }}</strong><span>{{ node.detail }}</span><small>{{ step.activeNodes.includes(node.id) ? '当前参与' : '本阶段未参与' }}</small>
        </div>
      </div>
      <div v-if="step.messages.length" class="flow-scroll" tabindex="0" aria-label="当前消息方向示意，可水平滚动；详细内容见下方消息列表">
        <svg :key="`${mode}-${stepIndex}`" class="flow" :viewBox="`0 0 720 ${step.messages.length * 94}`" role="presentation" aria-hidden="true">
          <defs>
            <marker :id="markerId" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" class="arrow-head" />
            </marker>
          </defs>
          <g v-for="(message, index) in step.messages" :key="index" :transform="`translate(0 ${index * 94})`">
            <rect class="endpoint" x="8" y="23" width="155" height="42" rx="7" />
            <rect class="endpoint" x="557" y="23" width="155" height="42" rx="7" />
            <text class="endpoint-text" x="85" y="49" text-anchor="middle">{{ label(message.from) }}</text>
            <text class="endpoint-text" x="635" y="49" text-anchor="middle">{{ label(message.to) }}</text>
            <path :class="['flow-line', networkClass(message.network)]" d="M 170 44 L 548 44" :marker-end="`url(#${markerId})`" />
            <text class="flow-label" x="359" y="31" text-anchor="middle">{{ message.network }}</text>
            <circle class="packet" r="6">
              <animateMotion path="M 170 44 L 548 44" dur="1.4s" begin="0s" fill="freeze" />
              <animate attributeName="opacity" values="1;1;0" keyTimes="0;0.85;1" dur="1.4s" begin="0s" fill="freeze" />
            </circle>
          </g>
        </svg>
      </div>
      <div class="messages">
        <strong>当前动作与方向</strong>
        <p v-if="!step.messages.length" class="empty">无消息：当前只观察初始状态。</p>
        <ul v-else>
          <li v-for="(message, index) in step.messages" :key="index">
            <span class="route">{{ label(message.from) }} <span aria-label="发送到">→</span> {{ label(message.to) }}</span>
            <span class="network">{{ message.network }}</span>
            <span class="payload">{{ message.payload }}</span>
          </li>
        </ul>
      </div>
    </div>
    <details class="assumptions" open>
      <summary>示意假设与边界</summary>
      <ul><li v-for="item in scenario.assumptions" :key="item">{{ item }}</li></ul>
    </details>
  </section>
</template>

<script setup>
import { computed, onUnmounted, ref, useId, watch } from 'vue'
import { STORAGE_SCENARIOS } from './networkStorageModel.mjs'

const props = defineProps({
  mode: { type: String, default: 'nfs', validator: value => Object.hasOwn(STORAGE_SCENARIOS, value) },
})
const markerId = `storage-arrow-${useId()}`
const scenario = computed(() => STORAGE_SCENARIOS[props.mode] || STORAGE_SCENARIOS.nfs)
function networkClass(network) {
  if (network.startsWith('cluster')) return 'cluster'
  if (network.startsWith('public')) return 'public'
  return 'local'
}
const stepIndex = ref(0)
const playing = ref(false)
const step = computed(() => scenario.value.steps[stepIndex.value])
let timer

function pause() {
  playing.value = false
  clearInterval(timer)
  timer = undefined
}
function reset() { pause(); stepIndex.value = 0 }
function goTo(index) {
  pause()
  stepIndex.value = Math.max(0, Math.min(scenario.value.steps.length - 1, index))
}
function toggle() {
  if (playing.value) return pause()
  if (stepIndex.value >= scenario.value.steps.length - 1) return
  playing.value = true
  timer = setInterval(() => {
    stepIndex.value++
    if (stepIndex.value >= scenario.value.steps.length - 1) pause()
  }, 2400)
}
function label(id) { return scenario.value.nodes.find(node => node.id === id)?.label ?? id }
watch(() => props.mode, reset, { flush: 'sync' })
onUnmounted(pause)
</script>

<style scoped>
.storage-demo { min-width: 0; margin: 1.2rem 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); overflow-wrap: anywhere; }
.network-legend, .assumptions { color: var(--vp-c-text-2); font-size: 13px; line-height: 1.6; }
.network-legend { margin: 6px 0; }
.controls, .steps { display: flex; flex-wrap: wrap; gap: 7px; margin: 12px 0; }
.controls button, .steps button { max-width: 100%; padding: 5px 10px; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); font-size: 13px; line-height: 1.4; text-align: left; cursor: pointer; }
.controls button:disabled { opacity: .45; cursor: default; }
.steps button.selected { border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); font-weight: 700; }
button:focus-visible, summary:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.stage { min-width: 0; padding: 12px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); }
.stage-heading { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; }
.stage-heading span { color: var(--vp-c-brand-1); font-size: 12px; font-weight: 700; }
.stage p { margin: 6px 0 12px; font-size: 13px; line-height: 1.6; }
.nodes { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 125px), 1fr)); gap: 8px; }
.node { min-width: 0; display: flex; flex-direction: column; gap: 2px; padding: 8px; border: 1px solid var(--vp-c-divider); border-radius: 6px; color: var(--vp-c-text-2); font-size: 12px; transition: border-color .25s, background-color .25s; }
.node.active { border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); color: var(--vp-c-text-1); }
.node small { font-size: 11px; }
.flow-scroll { max-width: 100%; min-width: 0; overflow-x: auto; margin-top: 12px; }
.flow-scroll:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.flow { display: block; width: 100%; min-width: 600px; height: auto; }
.endpoint { fill: var(--vp-c-bg-soft); stroke: var(--vp-c-divider); }
.endpoint-text { fill: var(--vp-c-text-1); font-size: 15px; font-weight: 700; }
.flow-line { fill: none; stroke: var(--vp-c-brand-1); stroke-width: 2.5; }
.flow-line.cluster { stroke: var(--vp-c-warning-1); stroke-dasharray: 9 5; }
.flow-line.local { stroke: var(--vp-c-text-2); stroke-dasharray: 2 5; }
.arrow-head { fill: var(--vp-c-text-1); }
.flow-label { fill: var(--vp-c-text-1); font-size: 13px; }
.packet { fill: var(--vp-c-brand-1); stroke: var(--vp-c-bg); stroke-width: 2; }
.messages { margin-top: 12px; font-size: 13px; }
.messages ul { list-style: none; padding: 0; margin: 6px 0 0; }
.messages li { display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: baseline; padding: 6px 0; border-top: 1px solid var(--vp-c-divider); }
.route { font-weight: 700; }
.network { padding: 1px 6px; border-radius: 4px; background: var(--vp-c-default-soft); font-size: 12px; }
.payload { flex: 1 1 200px; min-width: 0; }
.messages .empty { margin-bottom: 0; color: var(--vp-c-text-2); }
.assumptions { margin-top: 12px; }
.assumptions summary { cursor: pointer; font-weight: 600; }
.assumptions ul { margin: 6px 0 0; padding-left: 20px; }
@media (prefers-reduced-motion: reduce) { .node { transition: none; } .packet { display: none; } }
</style>
