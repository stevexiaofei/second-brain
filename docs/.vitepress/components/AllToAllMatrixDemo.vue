<template>
  <section class="a2a-demo" aria-label="All-to-All split 矩阵交互演示">
    <div class="a2a-head">
      <div><strong>All-to-All：发送矩阵与接收矩阵互为转置</strong><p>点击任一格，追踪“来源 rank → 目标 rank”的 dim-0 行数。</p></div>
      <button type="button" class="reset" @click="selected = { source: 1, target: 0 }">重置示例</button>
    </div>
    <div class="matrix-scroll">
      <table class="split-matrix">
        <caption>发送矩阵：行 = 发送方，列 = 接收方；单元格 = input_split_sizes[目标 rank]</caption>
        <thead><tr><th scope="col">发送方 ↓ / 接收方 →</th><th v-for="rank in ranks" :key="'h'+rank" scope="col">rank {{ rank }}</th></tr></thead>
        <tbody>
          <tr v-for="(row, source) in matrix" :key="source" :class="{ sourceRow: source === selected.source }">
            <th scope="row">rank {{ source }}</th>
            <td v-for="(count, target) in row" :key="source+'-'+target">
              <button type="button" class="cell" :class="cellClass(source, target)" :aria-label="`rank ${source} 发 ${count} 行给 rank ${target}`" :aria-pressed="isSelected(source,target)" @click="selected = { source, target }">{{ count }}</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div class="receive-panel">
        <strong>接收侧 output_split_sizes（按来源 rank 排列）</strong>
        <div v-for="target in ranks" :key="target" class="receive-row" :class="{ targetRow: target === selected.target }">
          <b>rank {{ target }}</b>
          <span v-for="source in ranks" :key="`${target}-${source}`" class="receive-count" :class="{ linked: isSelected(source,target) }">r{{ source }}:{{ matrix[source][target] }}</span>
          <small>← 列 {{ target }} 的转置</small>
        </div>
      </div>
    </div>
    <div class="flow-readout" aria-live="polite">
      <strong>rank {{ selected.source }} → rank {{ selected.target }}：{{ amount }} 行</strong>
      <span>发送方的 input_split_sizes[{{ selected.target }}] = {{ amount }}；接收方的 output_split_sizes[{{ selected.source }}] 也必须 = {{ amount }}。{{ payload }}</span>
    </div>
    <div class="matrix-foot"><span><i class="source-mark"></i>发送方向</span><span><i class="target-mark"></i>接收方向</span><span>全局 dim-0 守恒：输入 27 行 = 输出 27 行</span></div>
    <p class="a2a-note">单个 rank 发出与收到的总行数可以不同（如 rank 0 发 6、收 9）；必须逐对匹配的是“我发给你多少 = 你声明从我收多少”。</p>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'

const ranks = [0, 1, 2, 3]
const matrix = [
  [2, 2, 1, 1],
  [3, 2, 2, 2],
  [2, 1, 1, 1],
  [2, 2, 2, 1],
]
const selected = ref({ source: 1, target: 0 })
const amount = computed(() => matrix[selected.value.source][selected.value.target])
const payload = computed(() => {
  const { source, target } = selected.value
  return `因此 rank ${target} 的接收列表第 ${source} 项必须记录 ${amount.value}，接收到的数据段来自 rank ${source}。`
})
function isSelected(source, target) { return source === selected.value.source && target === selected.value.target }
function cellClass(source, target) { return { selected: isSelected(source,target), sourceHint: source === selected.value.source, targetHint: target === selected.value.target } }
</script>

<style scoped>
.a2a-demo { --surface:#fcfcfb; --card:#fff; --ink:#1f2328; --muted:#57606a; --line:#d0d7de; --soft:#f5f5f4; --source:#b85c2d; --target:#2878c8; margin:1rem 0 1.4rem; padding:14px 16px; border:1px solid var(--line); border-radius:10px; background:var(--surface); color:var(--ink); }
.dark .a2a-demo { --surface:#1a1a19; --card:#242423; --ink:#e6e6e3; --muted:#9d9d97; --line:#3d3d3b; --soft:#2b2b2a; --source:#e99468; --target:#68a9ed; }
.a2a-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
.a2a-head strong { font-size:14px; }
.a2a-head p,.a2a-note { margin:3px 0 0; color:var(--muted); font-size:12px; line-height:1.55; }
.reset { border:1px solid var(--line); border-radius:6px; padding:5px 9px; color:var(--ink); background:var(--card); white-space:nowrap; cursor:pointer; }
.matrix-scroll { display:flex; align-items:flex-start; gap:16px; overflow-x:auto; padding:12px 2px 8px; }
.split-matrix { flex:none; border-collapse:separate; border-spacing:3px; font-size:12px; }
.split-matrix caption { padding-bottom:6px; color:var(--muted); font-size:11px; text-align:left; }
.split-matrix th { padding:5px 7px; color:var(--muted); font-weight:600; white-space:nowrap; }
.split-matrix tbody th { text-align:left; }
.cell { width:42px; height:36px; border:1px solid var(--line); border-radius:5px; background:var(--card); color:var(--ink); font-weight:600; cursor:pointer; }
.cell.sourceHint { border-bottom:2px solid var(--source); }
.cell.targetHint { border-right:2px solid var(--target); }
.cell.selected { background:var(--target); border-color:var(--target); color:#fff; box-shadow:0 0 0 2px color-mix(in srgb, var(--target) 24%, transparent); }
.sourceRow th { color:var(--source); }
.receive-panel { min-width:225px; padding-top:21px; }
.receive-panel > strong { display:block; margin-bottom:8px; font-size:11px; color:var(--muted); }
.receive-row { display:flex; align-items:center; gap:5px; padding:5px 6px; border-radius:6px; font-size:11px; white-space:nowrap; }
.receive-row.targetRow { background:var(--soft); }
.receive-row b { min-width:45px; }
.receive-count { display:inline-flex; min-width:38px; height:23px; padding:0 3px; box-sizing:border-box; align-items:center; justify-content:center; border:1px solid var(--line); border-radius:4px; background:var(--card); }
.receive-count.linked { border:2px solid var(--target); color:var(--target); font-weight:700; }
.receive-row small { margin-left:4px; color:var(--muted); }
.flow-readout { display:grid; gap:3px; padding:9px 12px; border-radius:7px; background:var(--soft); font-size:12px; line-height:1.55; }
.flow-readout strong { font-size:13px; }
.matrix-foot { display:flex; flex-wrap:wrap; gap:12px; margin-top:9px; color:var(--muted); font-size:11px; }
.matrix-foot i { display:inline-block; width:9px; height:9px; margin-right:4px; border-radius:2px; vertical-align:-1px; }
.source-mark { background:var(--source); }.target-mark { background:var(--target); }
.a2a-note { margin-top:6px; }
@media (max-width:650px) { .matrix-scroll { display:block; } .receive-panel { padding-top:6px; } }
</style>
