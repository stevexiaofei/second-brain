<template>
  <section class="reducer-demo" aria-label="DDP Reducer 桶就绪与归约动画">
    <header class="demo-header">
      <div>
        <strong>Reducer 反向：梯度就绪 → 按序归约 → finalize</strong>
        <p>示例有 2 个 bucket；梯度按 W2 → W0 → W3 → W1 到达，让 Bucket 1 先 ready。</p>
      </div>
      <div class="controls" aria-label="动画控制">
        <button type="button" aria-label="上一步" title="上一步" @click="step(-1)">◀</button>
        <button type="button" :aria-label="playing ? '暂停' : '播放'" :title="playing ? '暂停' : '播放'" @click="toggle">{{ playing ? '⏸' : '▶' }}</button>
        <button type="button" aria-label="下一步" title="下一步" @click="step(1)">▶</button>
      </div>
    </header>

    <nav class="stage-tabs" aria-label="Reducer 动画阶段">
      <button v-for="(item, i) in stages" :key="item.short" type="button" :aria-current="i === stage ? 'step' : undefined" :class="{ active: i === stage }" @click="go(i)">{{ item.short }}</button>
    </nav>

    <div class="flow-viewport">
      <div class="flow-grid">
        <section class="panel event-panel" aria-label="梯度到达事件">
          <h4>Autograd 事件</h4>
          <p class="panel-caption">post-hook 告知 Reducer 梯度已就绪</p>
          <ol class="event-list">
            <li v-for="(event, i) in events" :key="event.param" :class="eventClass(i)">
              <span class="event-num">{{ i + 1 }}</span>
              <span class="event-text"><b>{{ event.param }}</b><small>{{ event.label }}</small></span>
              <span class="event-mark" :aria-label="i < eventCount ? '已到达' : '等待中'">{{ i < eventCount ? '✓' : '·' }}</span>
            </li>
          </ol>
          <button v-for="param in params" :key="param.name" type="button" class="param-chip" :class="{ picked: selectedParam === param.name }" @click="selectParam(param.name)">
            {{ param.name }} <small>bucket {{ param.bucket }}</small>
          </button>
          <p class="map-note" aria-live="polite">{{ selectedParam ? `${selectedParam} → variable_locators_ → Bucket ${paramBucket(selectedParam)}` : '点参数查看 variable_locators_ 的映射' }}</p>
        </section>

        <section class="panel bucket-panel" aria-label="Bucket pending 计数与归约状态">
          <h4>Bucket 状态</h4>
          <p class="panel-caption">pending = 本桶尚未就绪的梯度数</p>
          <article v-for="bucket in bucketViews" :key="bucket.id" class="bucket-card" :class="bucket.statusClass">
            <div class="bucket-top">
              <strong>Bucket {{ bucket.id }}</strong>
              <span class="status-pill">{{ bucket.statusText }}</span>
            </div>
            <div class="bucket-params">
              <span v-for="name in bucket.params" :key="name" class="bucket-param" :class="{ ready: isParamReady(name), current: currentParam === name }">{{ name }}<i>{{ isParamReady(name) ? '✓' : '…' }}</i></span>
            </div>
            <div class="pending-row"><span>pending</span><strong>{{ bucket.pending }}</strong><span class="pending-track"><i :style="{ width: `${(2 - bucket.pending) * 50}%` }"></i></span></div>
            <div v-if="bucket.communication" class="comm-state"><span class="pulse-dot"></span>all-reduce 已发起 · Future {{ bucket.futureState }}</div>
            <div v-else-if="bucket.waitingForTurn" class="wait-state">已 ready，但等待 Bucket {{ nextBucket }} 先归约</div>
          </article>
          <div class="order-strip" aria-live="polite">
            <span class="order-label">发起顺序</span>
            <b v-for="(id, i) in reductionOrder" :key="id" :class="{ reached: collectivesStarted }">B{{ id }}<i v-if="i < reductionOrder.length - 1">→</i></b>
            <span v-if="!collectivesStarted" class="order-pending">尚未触发</span>
          </div>
        </section>

        <section class="panel finalize-panel" aria-label="Reducer 顺序指针与 finalize 状态">
          <h4>顺序与收尾</h4>
          <div class="pointer-box">
            <small>next_bucket_</small>
            <strong>{{ nextBucketLabel }}</strong>
            <span>{{ nextBucketDescription }}</span>
          </div>
          <div class="future-box">
            <b>bucket.future_work</b>
            <div v-for="bucket in bucketViews" :key="bucket.id" class="future-row">
              <span>B{{ bucket.id }}</span><span :class="bucket.communication ? 'future-live' : ''">{{ bucket.futureState }}</span>
            </div>
          </div>
          <div class="finalize-box" :class="{ finalizing: stage >= 5, finished: stage === 6 }">
            <span class="finalize-icon">{{ stage === 6 ? '✓' : stage >= 5 ? '↻' : '…' }}</span>
            <div><b>{{ finalizeTitle }}</b><small>{{ finalizeDescription }}</small></div>
          </div>
        </section>
      </div>
    </div>

    <div class="explanation" aria-live="polite">
      <strong>{{ stages[stage].title }}</strong>
      <span>{{ stages[stage].description }}</span>
    </div>
    <footer class="legend">
      <span><i class="legend-swatch ready-swatch"></i>梯度 ready</span>
      <span><i class="legend-swatch wait-swatch"></i>等前序 bucket</span>
      <span><i class="legend-swatch comm-swatch"></i>通信 Future</span>
      <span>文本标签同样标注状态，不依赖颜色判断</span>
    </footer>
  </section>
</template>

<script setup>
import { computed, onUnmounted, ref } from 'vue'

const params = [
  { name: 'W0', bucket: 0 }, { name: 'W1', bucket: 0 },
  { name: 'W2', bucket: 1 }, { name: 'W3', bucket: 1 },
]
const events = [
  { param: 'W2', label: 'Bucket 1 · pending 2→1' },
  { param: 'W0', label: 'Bucket 0 · pending 2→1' },
  { param: 'W3', label: 'Bucket 1 · pending 1→0' },
  { param: 'W1', label: 'Bucket 0 · pending 1→0' },
]
const stages = [
  { short: '重置', title: '① prepare_for_backward：清点本轮梯度', description: 'Reducer 把 next_bucket_ 归零，并将各 bucket.pending 设为桶内参数数目。接下来 autograd hook 会逐个报告梯度 ready。' },
  { short: 'W2 ready', title: '② 收到 W2 梯度：Bucket 1 还没齐', description: 'W2 属于 Bucket 1。它的 pending 从 2 变为 1；桶里还有 W3 的梯度未就绪，因此不发起 all-reduce。' },
  { short: 'W0 ready', title: '③ 收到 W0 梯度：Bucket 0 还没齐', description: 'W0 属于 Bucket 0，pending 从 2 变为 1。此时两个桶都没有齐活，Reducer 继续让反向计算运行。' },
  { short: 'W3 ready', title: '④ Bucket 1 先 ready，但不能越序通信', description: 'W3 到达后 Bucket 1.pending 归零；然而 next_bucket_=0，Bucket 0 还未 ready，所以 Bucket 1 只能等待，不能先调用 all-reduce。' },
  { short: 'W1 ready', title: '⑤ Bucket 0 ready：顺序扫描并连续发起归约', description: 'W1 让 Bucket 0.pending 归零。Reducer 从 next_bucket_=0 开始，先发起 Bucket 0，再发现 Bucket 1 也已 ready，紧接着发起 Bucket 1；跨 rank 调用顺序保持一致。' },
  { short: '排队收尾', title: '⑥ 桶已发起通信：等待 autograd 引擎结束', description: '最后一个桶的归约已排队，Reducer 注册 Engine::queue_callback。它不会阻塞当前反向节点；回调要等反向图中其余计算也结束后才运行。' },
  { short: '写回 .grad', title: '⑦ finalize_backward：等 Future 并写回梯度', description: 'finalize_backward 等待各 bucket.future_work 完成，再把归约结果通过 bucket_views_out / unflatten 写回参数 .grad，最后复位标志，为下一迭代做准备。' },
]
const stage = ref(0)
const selectedParam = ref(null)
const playing = ref(false)
let timer
const eventCount = computed(() => Math.min(4, stage.value))
const currentParam = computed(() => stage.value >= 1 && stage.value <= 4 ? events[stage.value - 1].param : null)
const collectivesStarted = computed(() => stage.value >= 4)
const reductionOrder = computed(() => collectivesStarted.value ? [0, 1] : [])
const nextBucket = computed(() => stage.value >= 4 ? 2 : 0)
const nextBucketLabel = computed(() => stage.value >= 4 ? '2 / 2' : '0 / 2')
const nextBucketDescription = computed(() => stage.value >= 4 ? '所有 bucket 的归约都已发起' : '必须按 bucket 索引顺序推进')
const finalizeTitle = computed(() => stage.value === 6 ? 'finalize 完成' : stage.value >= 5 ? '等待 Engine callback' : '尚未进入 finalize')
const finalizeDescription = computed(() => stage.value === 6 ? '归约结果已写回各参数 .grad' : stage.value >= 5 ? '反向计算全部结束后，才等待 Future 并写回' : '最后一个 bucket ready 后才排入回调')
const bucketViews = computed(() => [0, 1].map(id => {
  const names = id === 0 ? ['W0', 'W1'] : ['W2', 'W3']
  const readyCount = names.filter(isParamReady).length
  const pending = 2 - readyCount
  const communication = stage.value >= 4
  const waitingForTurn = id === 1 && stage.value === 3
  const statusText = stage.value === 6 && communication ? '已完成' : communication ? '通信中' : waitingForTurn ? 'ready · 等顺序' : pending === 0 ? 'ready' : '收集梯度'
  return {
    id, params: names, pending, communication, waitingForTurn, statusText,
    statusClass: communication ? (stage.value === 6 ? 'done' : 'communicating') : waitingForTurn ? 'waiting' : pending === 0 ? 'ready' : '',
    futureState: stage.value === 6 ? 'completed' : communication ? 'in flight' : '未创建',
  }
}))
function isParamReady(name) { return events.slice(0, eventCount.value).some(event => event.param === name) }
function eventClass(index) { return { arrived: index < eventCount.value, current: events[index]?.param === currentParam.value } }
function paramBucket(name) { return params.find(param => param.name === name)?.bucket }
function selectParam(name) { selectedParam.value = selectedParam.value === name ? null : name; pause() }
function go(index) { stage.value = index; pause() }
function step(direction) { stage.value = (stage.value + direction + stages.length) % stages.length; pause() }
function toggle() {
  if (playing.value) return pause()
  playing.value = true
  timer = setInterval(() => { stage.value = (stage.value + 1) % stages.length }, 2400)
}
function pause() { playing.value = false; clearInterval(timer) }
onUnmounted(() => clearInterval(timer))
</script>

<style scoped>
.reducer-demo {
  --surface:#fcfcfb; --card:#fff; --ink:#1f2328; --muted:#57606a; --line:#d0d7de; --soft:#f5f5f4;
  --ready:#26734d; --ready-bg:#eaf5ee; --wait:#98551f; --wait-bg:#fff2df; --comm:#286db2; --comm-bg:#eaf3fc;
  margin:1rem 0 1.4rem; padding:14px 16px; border:1px solid var(--line); border-radius:10px; background:var(--surface); color:var(--ink);
}
.dark .reducer-demo { --surface:#1a1a19; --card:#242423; --ink:#e6e6e3; --muted:#9d9d97; --line:#3d3d3b; --soft:#2b2b2a; --ready:#8bd0a7; --ready-bg:#20382b; --wait:#eab27b; --wait-bg:#3c2c1d; --comm:#8fc3f4; --comm-bg:#1e3247; }
.demo-header { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
.demo-header strong { font-size:14px; }
.demo-header p,.panel-caption { margin:3px 0 0; color:var(--muted); font-size:11.5px; line-height:1.5; }
.controls { display:flex; flex:none; gap:4px; }
.controls button,.stage-tabs button { border:1px solid var(--line); border-radius:6px; padding:4px 8px; color:var(--ink); background:var(--card); cursor:pointer; }
.controls button { min-width:31px; }
.controls button:hover,.stage-tabs button:hover { border-color:var(--muted); }
.stage-tabs { display:flex; gap:5px; overflow-x:auto; padding:11px 0 9px; scrollbar-width:thin; }
.stage-tabs button { flex:none; font-size:11px; white-space:nowrap; }
.stage-tabs button.active { background:var(--soft); border-color:var(--muted); font-weight:700; }
.flow-viewport { overflow-x:auto; padding-bottom:3px; }
.flow-grid { display:grid; grid-template-columns:174px 1fr 192px; gap:9px; min-width:690px; }
.panel { min-width:0; padding:10px; border:1px solid var(--line); border-radius:8px; background:var(--card); }
.panel h4 { margin:0; color:var(--ink); font-size:12px; }
.event-list { display:grid; gap:5px; margin:9px 0 10px; padding:0; list-style:none; }
.event-list li { display:flex; align-items:center; gap:6px; min-height:35px; padding:4px 6px; border:1px solid var(--line); border-radius:6px; color:var(--muted); font-size:11px; }
.event-list li.arrived { color:var(--ink); background:var(--soft); }
.event-list li.current { border-color:var(--comm); box-shadow:0 0 0 1px var(--comm); }
.event-num { display:grid; width:17px; height:17px; flex:none; place-items:center; border-radius:50%; background:var(--soft); font-size:10px; }
.event-text { display:grid; min-width:0; }
.event-text small { color:var(--muted); font-size:9.5px; }
.event-mark { margin-left:auto; font-size:14px; }
.event-list li.arrived .event-mark { color:var(--ready); }
.param-chip { margin:0 3px 4px 0; padding:4px 6px; border:1px solid var(--line); border-radius:5px; background:var(--card); color:var(--ink); cursor:pointer; font-size:10px; }
.param-chip small { color:var(--muted); }
.param-chip.picked { border-color:var(--comm); box-shadow:0 0 0 1px var(--comm); }
.map-note { min-height:30px; margin:4px 0 0; color:var(--muted); font-size:10px; line-height:1.45; }
.bucket-card { margin-top:8px; padding:8px; border:1px solid var(--line); border-radius:7px; transition:border-color .2s, background .2s; }
.bucket-card.waiting { border-color:var(--wait); background:var(--wait-bg); }
.bucket-card.ready { border-color:var(--ready); }
.bucket-card.communicating { border-color:var(--comm); background:var(--comm-bg); }
.bucket-card.done { border-color:var(--ready); background:var(--ready-bg); }
.bucket-top { display:flex; justify-content:space-between; align-items:center; gap:5px; }
.bucket-top strong { font-size:11px; }
.status-pill { padding:2px 5px; border:1px solid var(--line); border-radius:10px; color:var(--muted); font-size:9px; white-space:nowrap; }
.waiting .status-pill { color:var(--wait); border-color:var(--wait); }
.ready .status-pill,.done .status-pill { color:var(--ready); border-color:var(--ready); }
.communicating .status-pill { color:var(--comm); border-color:var(--comm); }
.bucket-params { display:flex; gap:5px; margin:8px 0 7px; }
.bucket-param { display:flex; justify-content:space-between; align-items:center; gap:5px; min-width:55px; padding:4px 6px; border:1px solid var(--line); border-radius:5px; color:var(--muted); font-size:10px; }
.bucket-param.ready { color:var(--ready); border-color:var(--ready); }
.bucket-param.current { box-shadow:0 0 0 1px var(--comm); }
.bucket-param i { font-style:normal; }
.pending-row { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:10px; }
.pending-row strong { color:var(--ink); font-size:12px; }
.pending-track { height:5px; flex:1; overflow:hidden; border-radius:4px; background:var(--soft); }
.pending-track i { display:block; height:100%; border-radius:4px; background:var(--ready); transition:width .25s ease; }
.wait-state,.comm-state { margin-top:6px; font-size:9.5px; line-height:1.35; }
.wait-state { color:var(--wait); }
.comm-state { color:var(--comm); }
.pulse-dot { display:inline-block; width:7px; height:7px; margin-right:4px; border-radius:50%; background:var(--comm); animation:pulse 1s ease-in-out infinite alternate; }
.order-strip { display:flex; align-items:center; flex-wrap:wrap; gap:5px; margin-top:9px; padding:6px; border-radius:6px; background:var(--soft); font-size:10px; }
.order-label { color:var(--muted); }
.order-strip b { color:var(--muted); }
.order-strip b.reached { color:var(--comm); }
.order-strip b i { margin-left:5px; color:var(--muted); font-style:normal; }
.order-pending { color:var(--muted); font-size:9px; }
.pointer-box,.future-box { display:grid; gap:4px; padding:7px; border:1px solid var(--line); border-radius:6px; }
.pointer-box small,.future-box b { color:var(--muted); font-size:9.5px; }
.pointer-box strong { font-size:19px; line-height:1.1; }
.pointer-box span { color:var(--muted); font-size:9.5px; line-height:1.35; }
.future-box { margin-top:7px; }
.future-row { display:flex; justify-content:space-between; font-size:10px; }
.future-row span:last-child { color:var(--muted); }
.future-row .future-live { color:var(--comm); font-weight:600; }
.finalize-box { display:flex; gap:6px; align-items:flex-start; margin-top:7px; padding:7px; border:1px dashed var(--line); border-radius:6px; }
.finalize-box.finalizing { border-color:var(--comm); background:var(--comm-bg); }
.finalize-box.finished { border-color:var(--ready); background:var(--ready-bg); }
.finalize-icon { color:var(--muted); font-size:15px; }
.finished .finalize-icon { color:var(--ready); }
.finalize-box div { display:grid; gap:3px; }
.finalize-box b { font-size:9.5px; }
.finalize-box small { color:var(--muted); font-size:9px; line-height:1.35; }
.explanation { display:grid; gap:3px; margin-top:9px; padding:9px 12px; border-radius:7px; background:var(--soft); font-size:11.5px; line-height:1.55; }
.explanation strong { font-size:12px; }
.legend { display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; color:var(--muted); font-size:9.5px; }
.legend-swatch { display:inline-block; width:9px; height:9px; margin-right:4px; border:1px solid currentColor; border-radius:3px; vertical-align:-1px; }
.ready-swatch { color:var(--ready); background:var(--ready-bg); }.wait-swatch { color:var(--wait); background:var(--wait-bg); }.comm-swatch { color:var(--comm); background:var(--comm-bg); }
@keyframes pulse { to { opacity:.35; } }
@media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; } }
@media (max-width:600px) { .reducer-demo { padding:11px; } .flow-grid { min-width:690px; } .demo-header p { max-width:300px; } }
</style>
