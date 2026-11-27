// A displayed step is a teaching snapshot, not a global clock tick.
export const LAMPORT_SCENARIOS = {
  message: {
    title: '有消息：因果关系沿箭头传播',
    events: [
      { id: 'p1', process: 'P', type: 'local', x: 140, label: '修改笔记' },
      { id: 'p2', process: 'P', type: 'send', x: 250, label: '发送通知', message: 'm1' },
      { id: 'q1', process: 'Q', type: 'receive', x: 360, label: '收到通知', message: 'm1' },
      { id: 'q2', process: 'Q', type: 'send', x: 470, label: '发送回复', message: 'm2' },
      { id: 'p3', process: 'P', type: 'receive', x: 580, label: '收到回复', message: 'm2' },
    ],
    intro: 'P、Q 的计数器从 0 开始；只有事件或消息使编号变化，等待不会使编号增长。',
    steps: [
      { title: '初始状态', description: '两台机器各自记一个数，最初都是 0。它们不是秒表，也没有共同的全局计数器。' },
      { title: 'P 修改笔记', description: 'P 发生一次本地事件：先把自己的编号加 1，再给这个事件标记 1。Q 没有发生事件，仍是 0。' },
      { title: 'P 发送通知', description: '发送本身也是事件。P 从 1 加到 2，消息 m1 携带编号 2；这时 Q 还没收到，仍是 0。' },
      { title: 'Q 收到通知', description: 'Q 必须把“收到”排在“发送”后面：先取本地 0 与消息 2 的较大者，再加 1，得到 3。这不是照抄消息编号。' },
      { title: 'Q 发送回复', description: 'Q 的下一次事件是回复：从 3 加到 4，消息 m2 携带编号 4。P 尚未收到，仍是 2。' },
      { title: 'P 收到回复', description: 'P 比较自己的 2 与消息的 4，再加 1，得到 5。消息使 P 知道 Q 已经推进，因此它不能继续只记成 3。' },
    ],
    conclusion: '沿进程内顺序或消息箭头，编号严格增大。P 的“修改笔记”能沿这些箭头到达 Q 的“收到通知”，所以它们有因果先后。',
  },
  independent: {
    title: '没消息：编号有大小，也可能没有因果关系',
    events: [
      { id: 'p1', process: 'P', type: 'local', x: 180, label: '修改标题' },
      { id: 'q1', process: 'Q', type: 'local', x: 300, label: '修改字体' },
      { id: 'q2', process: 'Q', type: 'local', x: 450, label: '再改字号' },
    ],
    intro: '这一轮两台机器没有交换消息，也没有其他通信路径。按按钮逐个展示，不代表系统产生了跨机器的因果关系。',
    steps: [
      { title: '初始状态', description: '两个计数器都为 0。下面只观察独立的本地事件，不发送任何消息。' },
      { title: 'P 独立修改', description: 'P 修改标题，编号变为 1。Q 不知道这件事。' },
      { title: 'Q 独立修改', description: 'Q 修改字体，自己的编号也变为 1。P 和 Q 的编号可以相同：它们不是从一台中央机器领取全局序号。' },
      { title: 'Q 又修改一次', description: 'Q 再改字号，自己的编号变为 2。P 的事件编号是 1，Q 的新事件编号是 2，但它们之间没有消息路径：Q 并不知道 P 改过标题。' },
    ],
    conclusion: '反例出现了：P 的“修改标题”编号 1，小于 Q 的“再改字号”编号 2，但两者没有因果路径，属于因果上的并发。并发不要求在物理时间上同一瞬间发生。',
  },
}

export function getLamportState(mode, stepIndex) {
  const scenario = LAMPORT_SCENARIOS[mode]
  if (!scenario) throw new Error(`Unknown Lamport scenario: ${mode}`)
  if (!Number.isInteger(stepIndex) || stepIndex < 0 || stepIndex >= scenario.steps.length) throw new RangeError('Invalid Lamport step')
  const clocks = { P: 0, Q: 0 }
  const sent = {}
  const events = scenario.events.slice(0, stepIndex).map(event => {
    const before = clocks[event.process]
    let calculation
    if (event.type === 'receive') {
      const source = sent[event.message]
      if (!source) throw new Error('Receive must follow its send in the teaching trace')
      clocks[event.process] = Math.max(before, source.clock) + 1
      calculation = `取较大者（本地 ${before}，消息 ${source.clock}），再加 1 → ${clocks[event.process]}`
    } else {
      clocks[event.process]++
      calculation = `本地 ${before} 加 1 → ${clocks[event.process]}`
    }
    const result = { ...event, before, clock: clocks[event.process], calculation }
    if (event.type === 'send') sent[event.message] = result
    return result
  })
  const messages = Object.entries(sent).map(([id, source]) => ({
    id, source, target: events.find(event => event.type === 'receive' && event.message === id),
  }))
  return { clocks, events, messages, current: events.at(-1), ...scenario.steps[stepIndex] }
}
