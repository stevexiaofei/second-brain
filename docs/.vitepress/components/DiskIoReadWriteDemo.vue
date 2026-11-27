<template>
  <section class="io-demo" :aria-label="scenario.title" :data-io-mode="mode">
    <header class="demo-header">
      <p class="eyebrow">可暂停的机制演示 · 不是真实性能测量</p>
      <h3>{{ scenario.title }}</h3>
      <p class="goal">{{ scenario.goal }}</p>
      <label class="mode-select">切换教学主题
        <select :value="mode" @change="selectMode($event.target.value)">
          <option v-for="item in modes" :key="item.id" :value="item.id">{{ item.label }}</option>
        </select>
      </label>
    </header>

    <div class="branch-picker" role="group" aria-label="选择机制分支">
      <button v-for="item in scenario.variants" :key="item.id" type="button"
        :class="{ selected: variant === item.id }" :aria-pressed="variant === item.id" @click="chooseVariant(item.id)">
        {{ item.label }}
      </button>
    </div>
    <div v-if="mode === 'read'" class="read-options">
      <label>定位 API
        <select :value="api" :disabled="variant === 'readahead'" @change="changeOption('api', $event.target.value)">
          <option value="readv">readv：当前偏移 4096</option>
          <option value="preadv">preadv：显式 offset 8192</option>
        </select>
      </label>
      <label>后端
        <select :value="backend" @change="changeOption('backend', $event.target.value)">
          <option value="local">本地块设备</option><option value="network">网络文件系统</option>
        </select>
      </label>
      <p v-if="variant === 'readahead'">预读分支固定使用连续的 readv；示例窗口为 B + C，不代表实际内核窗口。</p>
    </div>

    <div class="controls" role="group" aria-label="动画播放控制">
      <button type="button" :disabled="stepIndex === 0" @click="goTo(stepIndex - 1)">上一步</button>
      <button type="button" @click="togglePlayback">{{ playing ? '暂停' : atEnd ? '重播' : '播放' }}</button>
      <button type="button" :disabled="atEnd" @click="goTo(stepIndex + 1)">下一步</button>
      <button type="button" @click="goTo(0)">重置</button>
      <label>步长
        <select v-model.number="delay" @change="pause">
          <option :value="1600">快：1.6 秒</option><option :value="3200">正常：3.2 秒</option><option :value="5500">细读：5.5 秒</option>
        </select>
      </label>
      <span class="step-count">第 {{ stepIndex + 1 }} / {{ scenario.steps.length }} 步</span>
    </div>
    <label class="step-jump">直接跳到阶段
      <select :value="stepIndex" @change="goTo(Number($event.target.value))">
        <option v-for="(item, i) in scenario.steps" :key="i" :value="i">{{ i + 1 }}. {{ item.title }}</option>
      </select>
    </label>

    <div class="stage-announcement" aria-live="polite" aria-atomic="true">
      <h4>{{ stepIndex + 1 }}. {{ frame.title }}</h4>
      <p>{{ frame.description }}</p>
    </div>
    <p class="code-label">当前 API / 伪代码（解释边界，不是可直接运行的完整程序）</p>
    <pre class="api-code"><code>{{ frame.code }}</code></pre>

    <div class="role-grid" aria-label="本机制参与者，点击可查看职责">
      <button v-for="node in scenario.nodes" :key="node.id" type="button" class="role-card"
        :class="{ active: activeRoles.has(node.id), inspected: inspected === node.id }"
        :aria-pressed="inspected === node.id" @click="inspected = inspected === node.id ? '' : node.id">
        <strong>{{ node.label }}</strong>
        <span>{{ activeRoles.has(node.id) ? '● 本步参与' : '○ 本步无消息' }}</span>
      </button>
    </div>
    <p v-if="inspectedNode" class="role-explanation"><strong>{{ inspectedNode.label }}：</strong>{{ inspectedNode.detail }}</p>

    <div class="events" aria-label="本步真实展示的方向，没有箭头的层不代表隐含访问">
      <p class="code-label">本步消息 / 数据流 · 箭头只表示本步动作</p>
      <p v-if="!frame.events.length" class="no-event">本步只观察状态或控制流，不画隐含的设备传输。</p>
      <article v-for="(message, index) in frame.events" :key="`${mode}-${variant}-${stepIndex}-${index}`" class="message-card">
        <div class="route-label"><strong>{{ nodeLabel(message.from) }}</strong><span>{{ message.from === message.to ? '内部处理' : '→' }}</span><strong>{{ nodeLabel(message.to) }}</strong></div>
        <div v-if="message.from !== message.to" class="message-track" aria-hidden="true">
          <span class="source-dot" /><span class="message-line" /><span class="target-arrow">›</span><span class="packet" />
        </div>
        <p><span class="message-kind">{{ message.kind }}</span> {{ message.payload }}</p>
      </article>
    </div>

    <div v-if="frame.state.pages.length" class="data-panel">
      <h4>客户端文件内容页</h4>
      <p>字母是可追踪的内容标签，不是字节数。框为空不等于文件逻辑范围不存在。</p>
      <div class="page-grid">
        <div v-for="item in frame.state.pages" :key="item.label" class="page-tile" :class="`page-${item.status}`">
          <span>{{ item.label }}</span><strong>{{ item.status === 'missing' ? '—' : item.value }}</strong><small>{{ pageLabels[item.status] }}</small>
        </div>
      </div>
    </div>
    <div v-if="frame.state.handles.length" class="table-scroll">
      <table class="handle-table">
        <caption>fd → 打开文件对象 → inode：共享的是哪一层？</caption>
        <thead><tr><th>fd</th><th>打开对象</th><th>该对象当前偏移 / 字节</th><th>inode</th></tr></thead>
        <tbody><tr v-for="handle in frame.state.handles" :key="handle.fd"><td>{{ handle.fd }}</td><td>{{ handle.object }}</td><td>{{ handle.offset }}</td><td>#{{ handle.inode }}</td></tr></tbody>
      </table>
    </div>
    <div v-if="mode === 'benchmark'" class="timing-notices">
      <p><strong>总吞吐计时：</strong>{{ frame.state.total }}</p>
      <p><strong>单次 latency：</strong>{{ frame.state.latency }}</p>
      <p>两个时钟是嵌套的代码边界，不代表两条硬件队列。</p>
    </div>

    <div class="table-scroll">
      <table class="state-table">
        <caption>{{ stepIndex ? '相对上一步的状态变化' : '初始观察状态' }} · 标记“变化”而非只靠颜色</caption>
        <thead><tr><th scope="col">对象 / 维度</th><th scope="col">上一步</th><th scope="col">当前状态</th></tr></thead>
        <tbody><tr v-for="row in stateRows" :key="row.id" :class="{ changed: row.changed }">
          <th scope="row">{{ row.label }}<span v-if="row.changed" class="change-label">变化</span></th><td>{{ row.before }}</td><td>{{ row.after }}</td>
        </tr></tbody>
      </table>
    </div>
    <p v-if="frame.note" class="frame-note"><strong>特别注意：</strong>{{ frame.note }}</p>
    <details class="assumptions"><summary>本组简化假设与边界（请勿把示意当作实现保证）</summary>
      <ul><li v-for="item in scenario.assumptions" :key="item">{{ item }}</li></ul>
    </details>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { getIoScenario, IO_DEMOS, PAGE_LABELS, STATE_FIELDS } from './diskIoReadWriteModel.mjs'

const props = defineProps({ mode: { type: String, default: 'open', validator: value => Object.hasOwn(IO_DEMOS, value) } })
const modes = Object.values(IO_DEMOS)
const mode = ref(Object.hasOwn(IO_DEMOS, props.mode) ? props.mode : 'open')
const variant = ref(IO_DEMOS[mode.value].variants[0].id)
const api = ref('readv')
const backend = ref('local')
const delay = ref(3200)
const stepIndex = ref(0)
const inspected = ref('')
const playing = ref(false)
const pageLabels = PAGE_LABELS
let timer

const scenario = computed(() => getIoScenario(mode.value, variant.value, { api: api.value, backend: backend.value }))
const frame = computed(() => scenario.value.steps[stepIndex.value])
const atEnd = computed(() => stepIndex.value === scenario.value.steps.length - 1)
const activeRoles = computed(() => new Set(frame.value.events.flatMap(message => [message.from, message.to])))
const inspectedNode = computed(() => scenario.value.nodes.find(node => node.id === inspected.value))
const stateRows = computed(() => {
  const previous = scenario.value.steps[Math.max(0, stepIndex.value - 1)].state
  return STATE_FIELDS.map(([id, label]) => ({ id, label, before: previous[id], after: frame.value.state[id], changed: previous[id] !== frame.value.state[id] }))
})
function nodeLabel(id) { return scenario.value.nodes.find(node => node.id === id)?.label || id }
function pause() { playing.value = false; clearInterval(timer); timer = undefined }
function goTo(index) { pause(); stepIndex.value = Math.max(0, Math.min(scenario.value.steps.length - 1, index)) }
function selectMode(value) {
  pause(); stepIndex.value = 0; inspected.value = ''
  mode.value = Object.hasOwn(IO_DEMOS, value) ? value : 'open'
  variant.value = IO_DEMOS[mode.value].variants[0].id
}
function chooseVariant(value) {
  pause(); stepIndex.value = 0; inspected.value = ''; variant.value = value
  if (mode.value === 'read' && value === 'readahead') api.value = 'readv'
}
function changeOption(name, value) {
  pause(); stepIndex.value = 0
  if (name === 'api') api.value = value
  else backend.value = value
}
function togglePlayback() {
  if (playing.value) return pause()
  if (atEnd.value) stepIndex.value = 0
  playing.value = true
  timer = setInterval(() => {
    if (atEnd.value) return pause()
    stepIndex.value += 1
    if (atEnd.value) pause()
  }, delay.value)
}
watch(() => props.mode, selectMode)
onBeforeUnmount(pause)
</script>

<style scoped>
.io-demo { margin: 1.5rem 0; padding: 18px; min-width: 0; max-width: 100%; border: 1px solid var(--vp-c-divider); border-radius: 12px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); overflow-wrap: anywhere; }
.io-demo h3 { margin: 4px 0 8px; font-size: 1.14rem; line-height: 1.45; }
.io-demo h4 { margin: 0 0 6px; font-size: 14px; }
.eyebrow, .code-label { margin: 0 0 6px; font-size: 11px; font-weight: 650; color: var(--vp-c-text-2); }
.goal { margin: 0 0 10px; font-size: 13px; color: var(--vp-c-text-2); }
.io-demo label { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; font-size: 12px; }
.io-demo select { min-width: 0; max-width: 100%; padding: 5px 8px; border: 1px solid var(--vp-c-divider); border-radius: 5px; background: var(--vp-c-bg); color: var(--vp-c-text-1); font: inherit; }
.mode-select select { flex: 1; }
.branch-picker, .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin-top: 13px; }
.io-demo button { padding: 6px 9px; max-width: 100%; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); font: inherit; font-size: 12px; cursor: pointer; text-align: left; }
.io-demo button.selected { border-width: 2px; border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); font-weight: 700; }
.io-demo button:disabled { opacity: .45; cursor: default; }
.io-demo :is(button, select, summary):focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 3px; }
.read-options { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
.read-options p { flex-basis: 100%; margin: 0; font-size: 12px; color: var(--vp-c-text-2); }
.controls { padding-top: 12px; border-top: 1px solid var(--vp-c-divider); }
.step-count { font-size: 12px; color: var(--vp-c-text-2); font-variant-numeric: tabular-nums; }
.step-jump { margin: 10px 0 14px; }
.step-jump select { width: 100%; }
.stage-announcement { padding: 12px; border-left: 3px solid var(--vp-c-brand-1); border-radius: 6px; background: var(--vp-c-bg); }
.stage-announcement p { margin: 5px 0 0; font-size: 13px; line-height: 1.7; }
.code-label { margin-top: 12px; }
.api-code { margin: 6px 0 12px; padding: 10px; overflow-x: auto; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-code-block-bg); }
.api-code code { padding: 0; border-radius: 0; background: none; color: var(--vp-c-text-1); font-size: 12px; white-space: pre; }
.role-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; }
.io-demo .role-card { display: flex; flex-direction: column; gap: 4px; min-width: 0; padding: 8px 10px; }
.role-card strong { font-size: 12px; }
.role-card span { color: var(--vp-c-text-2); font-size: 11px; }
.io-demo .role-card.active { border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); }
.io-demo .role-card.inspected { outline: 2px solid var(--vp-c-brand-1); outline-offset: 1px; }
.role-explanation { margin: 10px 0; padding: 8px; background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); font-size: 12px; line-height: 1.6; }
.no-event { margin: 6px 0; color: var(--vp-c-text-2); font-size: 12px; }
.message-card { margin: 7px 0; padding: 10px; border: 1px solid var(--vp-c-divider); border-radius: 7px; background: var(--vp-c-bg); }
.route-label { display: flex; justify-content: space-between; gap: 8px; align-items: center; font-size: 12px; }
.route-label > strong { flex: 1; }
.route-label > strong:last-child { text-align: right; }
.message-track { position: relative; height: 22px; margin: 4px 9px; }
.message-line { position: absolute; inset: 10px 0 auto; border-top: 2px solid var(--vp-c-text-2); }
.source-dot { position: absolute; left: -4px; top: 7px; width: 8px; height: 8px; border-radius: 50%; background: var(--vp-c-text-2); }
.target-arrow { position: absolute; right: -4px; top: -3px; line-height: 24px; font-size: 26px; color: var(--vp-c-text-1); }
.packet { position: absolute; width: 10px; height: 10px; top: 6px; left: 0; border-radius: 2px; background: var(--vp-c-brand-1); outline: 2px solid var(--vp-c-bg); animation: io-transfer 1.3s ease-in-out both; }
.message-card p { margin: 4px 0 0; font-size: 12px; line-height: 1.6; }
.message-kind { padding: 1px 5px; border: 1px solid var(--vp-c-divider); border-radius: 4px; background: var(--vp-c-bg-soft); font-size: 10px; }
.data-panel, .timing-notices { margin-top: 14px; padding: 10px; border: 1px solid var(--vp-c-divider); border-radius: 7px; background: var(--vp-c-bg); }
.data-panel > p, .timing-notices p { margin: 4px 0 8px; color: var(--vp-c-text-2); font-size: 12px; }
.page-grid { display: flex; flex-wrap: wrap; gap: 9px; }
.page-tile { display: flex; flex: 1 1 130px; flex-direction: column; align-items: center; gap: 6px; min-width: 0; padding: 9px; border: 2px solid var(--vp-c-text-2); border-radius: 7px; font-size: 11px; }
.page-tile strong { font-family: var(--vp-font-family-mono); font-size: 22px; }
.page-tile small { font-size: 11px; }
.page-missing { border-style: dashed; color: var(--vp-c-text-2); }
.page-dirty { border-style: dashed; border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); }
.page-writeback { border-style: double; border-width: 4px; }
.page-prefetched { border-style: dotted; }
.table-scroll { max-width: 100%; margin-top: 12px; overflow-x: auto; }
.io-demo table { display: table; width: 100%; margin: 0; border-collapse: collapse; font-size: 11px; }
.io-demo caption { margin-bottom: 6px; text-align: left; font-size: 11px; color: var(--vp-c-text-2); }
.io-demo :is(th, td) { padding: 7px; min-width: 95px; border: 1px solid var(--vp-c-divider); text-align: left; vertical-align: top; line-height: 1.6; }
.io-demo th { font-weight: 650; }
.io-demo tbody tr { background: var(--vp-c-bg); }
.io-demo tr.changed { background: var(--vp-c-brand-soft); }
.change-label { display: block; font-size: 10px; font-weight: 500; }
.frame-note { margin: 10px 0; padding: 8px; border: 1px dashed var(--vp-c-text-2); font-size: 12px; }
.assumptions { margin-top: 12px; color: var(--vp-c-text-2); font-size: 12px; line-height: 1.65; }
.assumptions summary { cursor: pointer; font-weight: 650; }
.assumptions ul { margin: 8px 0 0; padding-left: 18px; }
@keyframes io-transfer { 0% { left: 0; opacity: 0; } 15% { opacity: 1; } 85% { opacity: 1; } 100% { left: calc(100% - 10px); opacity: 0; } }
@media (max-width: 520px) { .io-demo { padding: 11px; } .role-grid { grid-template-columns: 1fr; } .mode-select select { flex: 1 1 100%; } .read-options > label { width: 100%; } .io-demo :is(th, td) { min-width: 110px; } }
@media (prefers-reduced-motion: reduce) { .packet { display: none; animation: none; } }
@media (forced-colors: active) { .page-tile, .role-card.active { outline: 2px solid Highlight; } }
</style>
