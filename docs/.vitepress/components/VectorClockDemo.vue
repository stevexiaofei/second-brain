<script setup>
import { computed, onUnmounted, ref, useId } from 'vue'
import { VECTOR_PARTICIPANTS, VECTOR_STEPS, compareVectorClocks, getVectorClockState } from './vectorClockModel.mjs'

const index = ref(0)
const playing = ref(false)
const inspected = ref(null)
const leftId = ref('a')
const rightId = ref('b')
const state = computed(() => getVectorClockState(index.value))
const detail = computed(() => state.value.events.find(event => event.id === inspected.value) || state.value.current)
const left = computed(() => state.value.events.find(event => event.id === leftId.value))
const right = computed(() => state.value.events.find(event => event.id === rightId.value))
const relation = computed(() => left.value && right.value ? compareVectorClocks(left.value.vector, right.value.vector) : null)
const end = computed(() => index.value === VECTOR_STEPS.length - 1)
const markerId = `vector-arrow-${useId()}`
const format = vector => `[${vector.join(', ')}]`
const y = process => process === 'A' ? 100 : 280
const symbol = (a, b) => a === b ? '=' : (a < b ? '<' : '>')
const path = message => `M ${message.source.x} ${y(message.source.process)} L ${message.target.x} ${y(message.target.process)}`
const relationText = computed(() => ({
  equal: '两边是同一事件的记录，没有严格先后。',
  before: '左边每一格都不大于右边，且至少一格更小：左事件在因果上先于右事件。',
  after: '右边每一格都不大于左边，且至少一格更小：右事件在因果上先于左事件。',
  concurrent: '有的格左边大，有的格右边大：双方互不包含对方的全部进展，两个事件因果并发。',
}[relation.value] || '先推进到两个事件都已发生，再比较。'))
let timer
function pause() { playing.value = false; clearInterval(timer); timer = undefined }
function go(value) {
  pause()
  inspected.value = null
  index.value = Math.max(0, Math.min(VECTOR_STEPS.length - 1, value))
  const ids = state.value.events.map(event => event.id)
  if (!ids.includes(leftId.value)) leftId.value = ids[0] || 'a'
  if (!ids.includes(rightId.value)) rightId.value = ids[1] || 'b'
}
function reset() { go(0); leftId.value = 'a'; rightId.value = 'b' }
function toggle() {
  if (playing.value) return pause()
  if (end.value) return
  inspected.value = null
  playing.value = true
  timer = setInterval(() => { index.value++; if (end.value) pause() }, 3800)
}
function inspect(id) { pause(); inspected.value = id }
function preset(step, a, b) { go(step); leftId.value = a; rightId.value = b }
function isCompared(event) { return relation.value !== null && (event.id === leftId.value || event.id === rightId.value) }
onUnmounted(pause)
</script>

<template>
  <section class="vector-demo" aria-label="向量时钟交互演示">
    <header><strong>一张表，分开记住两个人的进展</strong><p>固定列顺序：[甲的进展，乙的进展]。每个人持有自己的表；两个分量不是“读数 + 写数”。</p></header>
    <div class="controls" aria-label="向量时钟播放控制">
      <button type="button" @click="reset">重置</button>
      <button type="button" :disabled="index === 0" @click="go(index - 1)">上一步</button>
      <button type="button" :disabled="end && !playing" @click="toggle">{{ playing ? '暂停' : '播放' }}</button>
      <button type="button" :disabled="end" @click="go(index + 1)">下一步</button>
    </div>
    <nav class="steps" aria-label="选择向量时钟阶段">
      <button v-for="(step, n) in VECTOR_STEPS" :key="n" type="button" :aria-current="index === n ? 'step' : undefined" @click="go(n)">{{ n }}. {{ step.title }}</button>
    </nav>
    <div class="clock-cards" aria-label="两名参与者的当前进度表">
      <article v-for="person in VECTOR_PARTICIPANTS" :key="person.id" :class="{ active: state.current?.process === person.id }">
        <strong>{{ person.label }}现在知道的进展</strong>
        <div class="vector-cells">
          <div v-for="column in VECTOR_PARTICIPANTS" :key="column.id" :class="{ owned: person.id === column.id }">
            <span>{{ column.label }}的进展</span><b>{{ state.clocks[person.id][column.index] }}</b><small>{{ person.id === column.id ? '自己的格' : '从通信中获知' }}</small>
          </div>
        </div>
      </article>
    </div>
    <div class="explanation" aria-live="polite" aria-atomic="true">
      <strong>第 {{ index }} 步：{{ state.title }}</strong><p>{{ state.description }}</p>
    </div>
    <div class="timeline-scroll" tabindex="0" aria-label="两人的事件及消息时间线，可横向滚动，图下方有完整事件按钮">
      <svg :key="index" viewBox="0 0 740 360" class="timeline" aria-hidden="true">
        <defs><marker :id="markerId" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>
        <text x="20" y="30" class="axis-label">向右为展示顺序，不是物理时间；甲、乙之间只有消息箭头传播历史。</text>
        <g v-for="person in VECTOR_PARTICIPANTS" :key="person.id">
          <text x="20" :y="y(person.id) + 5" class="person-label">{{ person.label }}</text>
          <line x1="65" x2="710" :y1="y(person.id)" :y2="y(person.id)" class="lifeline" :marker-end="`url(#${markerId})`" />
        </g>
        <g v-for="message in state.messages" :key="message.id">
          <template v-if="message.target">
            <path :d="path(message)" class="message-line" :marker-end="`url(#${markerId})`" />
            <text :x="(message.source.x + message.target.x) / 2 + 8" y="195" class="message-label">{{ message.id }} {{ format(message.source.vector) }}</text>
            <circle v-if="state.current?.id === message.target.id" r="6" class="packet">
              <animateMotion :path="path(message)" dur="1.5s" fill="freeze" />
              <animate attributeName="opacity" values="1;1;0" keyTimes="0;0.9;1" dur="1.5s" fill="freeze" />
            </circle>
          </template>
          <text v-else :x="message.source.x + 8" :y="y(message.source.process) + (message.source.process === 'A' ? 50 : -50)" class="pending">{{ message.id }} 在途：{{ format(message.source.vector) }}</text>
        </g>
        <g v-for="event in state.events" :key="event.id" :class="['event', { current: state.current?.id === event.id, compared: isCompared(event), inspected: detail?.id === event.id }]">
          <rect :x="event.x - 37" :y="y(event.process) - 19" width="74" height="38" rx="8" />
          <text :x="event.x" :y="y(event.process) + 5" text-anchor="middle" class="event-vector">{{ format(event.vector) }}</text>
          <text :x="event.x" :y="y(event.process) + (event.process === 'A' ? -33 : 43)" text-anchor="middle" class="event-label">{{ event.label }}</text>
        </g>
        <text v-if="index > 0 && index < 3" x="130" y="195" class="axis-label">此时没有消息；点击按钮的先后不等于因果先后。</text>
        <text x="20" y="347" class="axis-label">事件上的向量是当时的快照，收到新消息也不会改写早先事件的向量。</text>
      </svg>
    </div>
    <div class="event-buttons" aria-label="查看已发生事件的计算">
      <span v-if="!state.events.length">点“下一步”，先看甲如何更新自己的格。</span>
      <button v-for="event in state.events" :key="event.id" type="button" :aria-pressed="detail?.id === event.id" @click="inspect(event.id)">{{ event.name }} {{ format(event.vector) }}</button>
    </div>
    <section v-if="detail" class="calculation" aria-label="逐格更新过程">
      <strong>{{ detail.name }}：这一张表怎么得到？</strong>
      <p>事件前本地表：<b>{{ format(detail.before) }}</b><template v-if="detail.incoming">；收到消息：<b>{{ format(detail.incoming) }}</b></template></p>
      <div class="calculation-rows">
        <div v-for="person in VECTOR_PARTICIPANTS" :key="person.id" class="calculation-row" :class="{ owned: person.index === detail.own }">
          <strong>{{ person.label }}的格</strong>
          <span v-if="detail.incoming">取较大值（本地 {{ detail.before[person.index] }}，消息 {{ detail.incoming[person.index] }}）→ {{ detail.merged[person.index] }}</span>
          <span v-else>原来 {{ detail.before[person.index] }}</span>
          <span>{{ person.index === detail.own ? '这是自己的格，再加 1' : '不是自己的格，不自增' }}</span>
          <b>结果 {{ detail.vector[person.index] }}</b>
        </div>
      </div>
      <p v-if="detail.merged">中间合并 {{ format(detail.merged) }} → 记录本次接收事件 → 最终 <b>{{ format(detail.vector) }}</b></p>
      <p v-else>最终 <b>{{ format(detail.vector) }}</b><template v-if="detail.type === 'send'">，以快照形式附在消息中。</template></p>
    </section>
    <section class="compare-panel" aria-label="比较两个事件的向量">
      <strong>动手比较：谁包含谁的历史？</strong>
      <div class="presets">
        <button type="button" @click="preset(2, 'a', 'b')">看两次独立修改</button>
        <button type="button" @click="preset(4, 'a', 'r1')">看收到消息之后</button>
        <button type="button" @click="preset(6, 'a', 'b')">通信后重看原修改</button>
      </div>
      <div v-if="state.events.length >= 2" class="selectors">
        <label>左事件<select v-model="leftId" @change="pause"><option v-for="event in state.events" :key="event.id" :value="event.id">{{ event.name }} {{ format(event.vector) }}</option></select></label>
        <label>右事件<select v-model="rightId" @change="pause"><option v-for="event in state.events" :key="event.id" :value="event.id">{{ event.name }} {{ format(event.vector) }}</option></select></label>
      </div>
      <div v-if="relation" class="comparison" aria-live="polite" aria-atomic="true">
        <div class="comparison-cells"><span v-for="person in VECTOR_PARTICIPANTS" :key="person.id">{{ person.label }}这一格：<b>{{ left.vector[person.index] }} {{ symbol(left.vector[person.index], right.vector[person.index]) }} {{ right.vector[person.index] }}</b></span></div>
        <p><strong>{{ relation === 'concurrent' ? '因果并发' : relation === 'equal' ? '同一事件' : '有因果先后' }}</strong>：{{ relationText }}</p>
        <p v-if="relation === 'concurrent'">不是把两格相加，也不是只看第一格。只有“一边所有格都不大于另一边，且至少一格更小”，才有严格的因果先后。</p>
        <p v-if="index >= 4 && leftId === 'a' && rightId === 'b'">即使已经通信，最初两次修改的向量仍是 [1,0] 与 [0,1]。后来知道彼此，不会让过去的并发变成先后，也不会自动决定会议该选周日还是周一。</p>
      </div>
      <p v-else>先发生两个事件，或点击上面的示例按钮，再逐格比较。</p>
    </section>
    <p class="boundary">教学假设：甲、乙是固定的两个进程，所有相关事件与消息都遵守这套规则，没有遗漏的通信路径；本地、发送、接收各计一次事件。比较的是事件快照，不是随意填写的数组；因果并发不要求物理上同时。动画只传播历史，不执行业务冲突合并。</p>
  </section>
</template>

<style scoped>
.vector-demo { min-width: 0; margin: 1.3rem 0; padding: 16px; border: 1px solid var(--vp-c-divider); border-radius: 12px; color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); overflow-wrap: anywhere; }
header p, .boundary { font-size: 13px; line-height: 1.7; color: var(--vp-c-text-2); }
.controls, .steps, .event-buttons, .presets { display: flex; flex-wrap: wrap; gap: 7px; margin: 12px 0; }
button, select { max-width: 100%; border: 1px solid var(--vp-c-divider); border-radius: 6px; padding: 6px 9px; color: var(--vp-c-text-1); background: var(--vp-c-bg); font-size: 13px; line-height: 1.5; }
button { text-align: left; cursor: pointer; }
button[aria-current="step"], button[aria-pressed="true"] { background: var(--vp-c-brand-soft); border-color: var(--vp-c-brand-1); font-weight: 700; }
button:disabled { opacity: .45; cursor: default; }
button:focus-visible, select:focus-visible, .timeline-scroll:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.clock-cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.clock-cards article { min-width: 0; padding: 12px; background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; font-size: 13px; }
.clock-cards article.active { border-color: var(--vp-c-brand-1); }
.vector-cells { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; margin-top: 9px; }
.vector-cells > div { display: flex; min-width: 0; flex-direction: column; align-items: center; gap: 2px; padding: 6px 3px; border: 1px solid var(--vp-c-divider); border-radius: 5px; }
.vector-cells b { font-size: 25px; }
.vector-cells small { font-size: 10px; color: var(--vp-c-text-2); text-align: center; }
.vector-cells .owned { border-style: dashed; border-color: var(--vp-c-brand-1); }
.explanation, .calculation, .compare-panel { margin-top: 12px; padding: 12px; background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; font-size: 14px; line-height: 1.7; }
.explanation p, .calculation p, .comparison p { margin: 6px 0 0; }
.timeline-scroll { max-width: 100%; margin-top: 12px; overflow-x: auto; background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; }
.timeline { display: block; width: 100%; min-width: 700px; height: auto; }
.timeline marker path { fill: var(--vp-c-text-2); }
.lifeline { stroke: var(--vp-c-text-2); stroke-width: 1.5; }
.axis-label, .pending { fill: var(--vp-c-text-2); font-size: 13px; }
.person-label { fill: var(--vp-c-text-1); font-size: 20px; font-weight: 700; }
.message-line { fill: none; stroke: var(--vp-c-brand-1); stroke-width: 2.5; stroke-dasharray: 6 3; }
.message-label { fill: var(--vp-c-text-1); font-size: 13px; paint-order: stroke; stroke: var(--vp-c-bg); stroke-width: 5px; stroke-linejoin: round; }
.event rect { fill: var(--vp-c-bg-soft); stroke: var(--vp-c-text-2); stroke-width: 1.5; }
.event.current rect, .event.inspected rect { fill: var(--vp-c-brand-soft); stroke: var(--vp-c-brand-1); stroke-width: 2.5; }
.event.compared rect { stroke-dasharray: 4 2; }
.event-vector { fill: var(--vp-c-text-1); font-size: 14px; font-weight: 700; }
.event-label { fill: var(--vp-c-text-1); font-size: 13px; }
.packet { fill: var(--vp-c-brand-1); stroke: var(--vp-c-bg); stroke-width: 2; }
.event-buttons { font-size: 13px; }
.calculation-rows { display: grid; gap: 6px; margin: 10px 0; }
.calculation-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 5px 12px; padding: 8px; border: 1px solid var(--vp-c-divider); border-radius: 5px; font-size: 13px; }
.calculation-row.owned { border-left: 3px solid var(--vp-c-brand-1); }
.selectors { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.selectors label { min-width: 0; display: flex; flex-direction: column; gap: 4px; font-size: 13px; }
select { width: 100%; }
.comparison { margin-top: 12px; padding: 10px; border-left: 3px solid var(--vp-c-brand-1); background: var(--vp-c-brand-soft); }
.comparison-cells { display: flex; flex-wrap: wrap; gap: 6px 18px; }
@media (max-width: 450px) { .clock-cards, .selectors { grid-template-columns: minmax(0, 1fr); } }
@media (prefers-reduced-motion: reduce) { .packet { display: none; } }
</style>
