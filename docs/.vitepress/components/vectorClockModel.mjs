// Fixed participants: component 0 belongs to A (甲), component 1 to B (乙).
// Every local/send/receive event increments the owner's component once.
export const VECTOR_PARTICIPANTS = [
  { id: 'A', label: '甲', index: 0 },
  { id: 'B', label: '乙', index: 1 },
]

export const VECTOR_EVENTS = [
  { id: 'a', process: 'A', type: 'local', x: 110, label: '改成周日', name: '甲改成周日' },
  { id: 'b', process: 'B', type: 'local', x: 210, label: '改成周一', name: '乙独立改成周一' },
  { id: 's1', process: 'A', type: 'send', x: 310, label: '发送通知', name: '甲发送通知', message: 'm1' },
  { id: 'r1', process: 'B', type: 'receive', x: 410, label: '收到通知', name: '乙收到通知', message: 'm1' },
  { id: 's2', process: 'B', type: 'send', x: 510, label: '发送回复', name: '乙发送回复', message: 'm2' },
  { id: 'r2', process: 'A', type: 'receive', x: 610, label: '收到回复', name: '甲收到回复', message: 'm2' },
]

export const VECTOR_STEPS = [
  { title: '各有一张进度表', description: '甲有自己的 [0,0]，乙也有自己的 [0,0]。不是两个人共用同一个数组：第一格固定记甲的事件进展，第二格固定记乙的事件进展。' },
  { title: '甲先独立修改', description: '甲把会议改成周日，只增加属于甲的第一格，得到 [1,0]。乙没有收到消息，仍是 [0,0]；甲的第二格为 0，也只是甲尚不知道乙有新事件。' },
  { title: '乙也独立修改', description: '乙没收到甲的消息，独立改成周一，只增加自己的第二格，得到 [0,1]。甲仍是 [1,0]。两份记录各领先一格，不能判定哪次修改包含另一方的历史。' },
  { title: '甲发送整张表', description: '发送也是一次事件，甲先把第一格从 1 加到 2，再把 [2,0] 的副本放进消息 m1。乙尚未收到，自己的表仍是 [0,1]。消息中的数组不会跟着甲后续的变化而改变。' },
  { title: '乙先合并，再记接收', description: '乙拿本地 [0,1] 与消息 [2,0] 逐格取较大值，得到中间结果 [2,1]；再把自己的第二格加 1，给这次接收事件编号 [2,2]。第一格的 2 是学到甲的历史，不是乙做了两次甲的事件。' },
  { title: '乙把已知历史带回去', description: '乙发送回复时只增加自己的第二格：[2,2] → [2,3]。回复 m2 携带 [2,3]，包括乙已经获知的甲的进展和乙自己的进展。甲还没收到，仍是 [2,0]。' },
  { title: '甲也学到乙的历史', description: '甲用本地 [2,0] 与回复 [2,3] 逐格取较大值，得到 [2,3]；再增加自己的第一格，给接收事件编号 [3,3]。现在彼此传播了历史，但最初“周日”和“周一”两次修改仍然是并发的。' },
]

export function compareVectorClocks(left, right) {
  if (left.length !== right.length || left.length === 0) throw new Error('Vector dimensions must match')
  const noGreater = left.every((value, i) => value <= right[i])
  const noSmaller = left.every((value, i) => value >= right[i])
  if (noGreater && noSmaller) return 'equal'
  if (noGreater) return 'before'
  if (noSmaller) return 'after'
  return 'concurrent'
}

export function getVectorClockState(step) {
  if (!Number.isInteger(step) || step < 0 || step >= VECTOR_STEPS.length) throw new RangeError('Invalid vector clock step')
  const clocks = { A: [0, 0], B: [0, 0] }
  const last = {}
  const sent = {}
  const events = VECTOR_EVENTS.slice(0, step).map(event => {
    const own = event.process === 'A' ? 0 : 1
    const before = [...clocks[event.process]]
    const causes = new Set()
    const predecessor = last[event.process]
    if (predecessor) {
      causes.add(predecessor.id)
      for (const id of predecessor.causes) causes.add(id)
    }
    let incoming = null
    let merged = null
    if (event.type === 'receive') {
      const source = sent[event.message]
      if (!source) throw new Error('A receive needs a previously sent message')
      incoming = [...source.vector]
      merged = before.map((value, i) => Math.max(value, incoming[i]))
      clocks[event.process] = [...merged]
      causes.add(source.id)
      for (const id of source.causes) causes.add(id)
    }
    clocks[event.process][own]++
    const result = { ...event, own, before, incoming, merged, vector: [...clocks[event.process]], causes: [...causes] }
    last[event.process] = result
    if (event.type === 'send') sent[event.message] = result
    return result
  })
  const messages = Object.entries(sent).map(([id, source]) => ({ id, source, target: events.find(event => event.type === 'receive' && event.message === id) }))
  return { ...VECTOR_STEPS[step], clocks, events, messages, current: events.at(-1) }
}
