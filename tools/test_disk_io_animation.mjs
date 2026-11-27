// Run: node --test tools/test_disk_io_animation.mjs
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc'
import { getIoScenario, IO_DEMOS, PAGE_LABELS, STATE_FIELDS } from '../docs/.vitepress/components/diskIoReadWriteModel.mjs'

const end = scenario => scenario.steps.at(-1).state
const findStep = (scenario, title) => {
  const step = scenario.steps.find(item => item.title.includes(title))
  assert.ok(step, `Missing step: ${title}`)
  return step
}

for (const [mode, base] of Object.entries(IO_DEMOS)) {
  for (const variant of base.variants) {
    test(`complete independent snapshots: ${mode}/${variant.id}`, () => {
      const scenario = getIoScenario(mode, variant.id)
      const ids = new Set(scenario.nodes.map(item => item.id))
      assert.ok(scenario.steps.length >= 4)
      for (const [i, step] of scenario.steps.entries()) {
        assert.ok(step.title && step.description && step.code, `frame ${i}`)
        for (const [field] of STATE_FIELDS) assert.equal(typeof step.state[field], 'string', `frame ${i}: ${field}`)
        for (const e of step.events) {
          assert.ok(ids.has(e.from) && ids.has(e.to), `${mode}/${variant.id}/${i}: ${e.from} -> ${e.to}`)
          assert.ok(e.payload && e.kind)
        }
        for (const p of step.state.pages) assert.ok(Object.hasOwn(PAGE_LABELS, p.status))
        if (i) {
          assert.notStrictEqual(step.state, scenario.steps[i - 1].state)
          assert.notStrictEqual(step.state.pages, scenario.steps[i - 1].state.pages)
        }
      }
      assert.deepEqual(scenario, getIoScenario(mode, variant.id))
    })
  }
}

test('open does not expose a returned fd before its allocation/return step', () => {
  const scenario = getIoScenario('open')
  const before = findStep(scenario, '建立打开对象 A')
  assert.equal(before.state.handles.length, 0)
  assert.deepEqual(findStep(scenario, '返回 fd 3').state.handles, [{ fd: 3, object: 'A', offset: 0, inode: 100 }])
})

test('independent open shares inode but not position; dup shares open object and position', () => {
  const independent = end(getIoScenario('open', 'independent')).handles
  const duplicated = end(getIoScenario('open', 'dup')).handles
  assert.deepEqual(independent.map(h => h.inode), [100, 100])
  assert.deepEqual(independent.map(h => h.object), ['A', 'B'])
  assert.deepEqual(independent.map(h => h.offset), [4096, 0])
  assert.deepEqual(duplicated.map(h => h.object), ['A', 'A'])
  assert.deepEqual(duplicated.map(h => h.offset), [4096, 4096])
})

for (const backend of ['local', 'network']) {
  for (const api of ['readv', 'preadv']) {
    for (const variant of ['hit', 'miss', 'readahead']) {
      test(`read routing and offsets: ${variant}/${api}/${backend}`, () => {
        const scenario = getIoScenario('read', variant, { api, backend })
        const messages = scenario.steps.flatMap(step => step.events)
        const ids = new Set(scenario.nodes.map(item => item.id))
        for (const message of messages) assert.ok(ids.has(message.from) && ids.has(message.to))
        if (variant === 'hit') {
          assert.ok(!messages.some(e => ['queue', 'device', 'server'].includes(e.to)))
          assert.match(end(scenario).backend, /没有后端数据读/)
        } else {
          assert.ok(messages.some(e => e.to === (backend === 'local' ? 'device' : 'server')))
          const fill = findStep(scenario, variant === 'readahead' ? '预读页填入' : '填充客户端')
          assert.match(fill.state.user, /0 \/ 4096/)
        }
        assert.ok(messages.some(e => e.from === 'cache' && e.to === 'app' && e.kind === '数据复制'))
        if (variant === 'readahead') {
          assert.equal(end(scenario).offset, '当前偏移 = 12288')
          assert.match(end(scenario).backend, /没有新增后端数据请求/)
          assert.equal(findStep(scenario, '预读页填入').state.pages[1].status, 'prefetched')
        } else if (api === 'preadv') {
          assert.equal(end(scenario).offset, '当前偏移 = 4096（不变）')
          assert.match(end(scenario).user, /内容 C/)
        } else {
          assert.equal(end(scenario).offset, '当前偏移 = 8192')
          assert.match(end(scenario).user, /内容 B/)
        }
      })
    }
  }
}

test('write may return dirty data without a new persistence guarantee', () => {
  const scenario = getIoScenario('write', 'batch')
  const returned = findStep(scenario, 'pwrite 0 返回')
  assert.equal(returned.state.pages[0].status, 'dirty')
  assert.match(returned.state.durability, /上次同步记录：A0 \/ A1/)
  assert.match(returned.state.durability, /覆盖后尚未确认同步：块 0/)
  assert.match(returned.state.durability, /不保证完整旧\/新版本/)
  assert.ok(scenario.steps.findIndex(s => s.title.includes('pwrite 1 返回')) < scenario.steps.findIndex(s => s.title.includes('调用 fsync')))
  assert.ok(end(scenario).pages.every(p => p.status === 'clean'))
  assert.match(end(scenario).durability, /B0 \/ B1/)
})

test('per-block sync confirms B0 before the second write; pressure branch blocks before acceptance', () => {
  const each = getIoScenario('write', 'each')
  assert.equal(each.steps.filter(s => s.title.startsWith('调用 fsync')).length, 2)
  const firstSync = findStep(each, 'fsync 成功返回')
  assert.match(firstSync.state.durability, /B0 \/ A1/)
  assert.match(firstSync.state.durability, /当前版本均已同步/)
  assert.match(findStep(each, 'pwrite 1 返回').state.durability, /覆盖后尚未确认同步：块 1/)
  const pressure = getIoScenario('write', 'pressure')
  const blocked = findStep(pressure, '脏页压力')
  assert.equal(blocked.state.pages[0].status, 'dirty')
  assert.equal(blocked.state.pages[1].status, 'missing')
  assert.match(blocked.state.app, /尚未返回/)
})

test('fsync does not evict pages; DONTNEED has two valid clean-page outcomes', () => {
  for (const variant of ['evicted', 'retained']) {
    const scenario = getIoScenario('sync', variant)
    const synced = findStep(scenario, 'fsync 成功')
    assert.equal(synced.state.pages[0].status, 'clean')
    assert.match(synced.state.durability, /内容 B/)
    const advised = findStep(scenario, 'DONTNEED：回收')
    assert.equal(advised.state.pages[0].status, variant === 'evicted' ? 'missing' : 'clean')
    assert.equal(advised.state.durability, synced.state.durability)
    assert.match(advised.state.backend, /未被该建议清理/)
  }
})

test('dirty-page advice and close-only branches do not invent a successful fsync', () => {
  const dirty = getIoScenario('sync', 'dirty')
  assert.equal(end(dirty).pages[0].status, 'dirty')
  assert.match(end(dirty).durability, /B 无新增同步保证/)
  assert.ok(!dirty.steps.some(s => s.title.startsWith('fsync 成功')))
  const close = getIoScenario('sync', 'close')
  assert.match(end(close).durability, /未获得显式 fsync 保证/)
  assert.equal(end(close).pages[0].status, 'dirty')
})

for (const variant of IO_DEMOS.benchmark.variants) {
  test(`benchmark timing follows script control flow: ${variant.id}`, () => {
    const scenario = getIoScenario('benchmark', variant.id)
    const isRead = ['seq_read', 'rand_read_4k'].includes(variant.id)
    const closeIndex = scenario.steps.findIndex(s => s.code === 'finally: os.close(fd)')
    const returnIndex = scenario.steps.findIndex(s => s.code.startsWith('return '))
    assert.ok(closeIndex >= 0 && returnIndex >= 0)
    assert.equal(closeIndex > returnIndex, isRead)
    assert.equal(findStep(scenario, '开始总计时').state.total, '计时中')
    assert.match(end(scenario).total, /已捕获/)
    const finalSyncs = scenario.steps.filter(s => s.code === 'os.fsync(fd)  # final')
    assert.equal(finalSyncs.length, ['seq_write_buf', 'seq_write_sync', 'rand_write_4k_buf'].includes(variant.id) ? 1 : 0)
    if (variant.id.startsWith('rand_')) {
      const latencyIndex = scenario.steps.findIndex(s => s.code.startsWith('lat.append('))
      const opIndex = scenario.steps.findIndex(s => s.code.startsWith(variant.id.includes('read') ? 'os.preadv' : 'os.pwrite'))
      assert.ok(opIndex < latencyIndex)
      if (variant.id === 'rand_write_4k_sync') {
        const fsyncIndex = scenario.steps.findIndex(s => s.code === 'os.fsync(fd)')
        assert.ok(opIndex < fsyncIndex && fsyncIndex < latencyIndex)
      }
      if (variant.id === 'rand_write_4k_buf') assert.ok(latencyIndex < scenario.steps.findIndex(s => s.code === 'os.fsync(fd)  # final'))
    }
  })
}

test('snapshots survive mutation of another step and invalid options fall back safely', () => {
  const scenario = getIoScenario('write', 'batch')
  const initial = structuredClone(scenario.steps[0])
  scenario.steps[2].state.pages[0].value = 'mutated'
  assert.deepEqual(scenario.steps[0], initial)
  assert.deepEqual(getIoScenario('unknown', 'unknown'), getIoScenario('open', 'independent'))
})

test('Vue SFC template and script compile with matching binding metadata', () => {
  const filename = new URL('../docs/.vitepress/components/DiskIoReadWriteDemo.vue', import.meta.url).pathname
  const source = readFileSync(new URL('../docs/.vitepress/components/DiskIoReadWriteDemo.vue', import.meta.url), 'utf8')
  const { descriptor, errors } = parse(source, { filename })
  assert.deepEqual(errors, [])
  const script = compileScript(descriptor, { id: 'disk-io-test' })
  const template = compileTemplate({ source: descriptor.template.content, filename, id: 'disk-io-test', compilerOptions: { bindingMetadata: script.bindings } })
  assert.deepEqual(template.errors, [])
})
