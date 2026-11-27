// Deterministic teaching snapshots, not timings or a simulation of a particular kernel.
// Every frame is independent: seeking backwards never has to undo live I/O.
const role = (id, label, detail) => ({ id, label, detail })
const ROLES = {
  app: role('app', '应用 / 用户缓冲区', 'bytes / bytearray 属于进程；与内核 page cache 是两份不同的内存。'),
  path: role('path', '路径与挂载解析', '沿目录分量查找；挂载关系决定具体文件系统。'),
  dentry: role('dentry', 'dentry / 目录查找', '内核中的父目录、名字和目标对象关系，不是直接写盘的目录记录。'),
  inode: role('inode', 'inode / 文件布局', '描述身份、权限、大小和布局；不是每次读写都重新解析路径。'),
  file: role('file', '打开文件对象', '记录当前偏移和打开状态；独立 open 与 dup 的共享关系不同。'),
  fd: role('fd', '进程 fd 表', 'fd 是进程内整数索引，不是 inode 号或磁盘地址。'),
  syscall: role('syscall', '系统调用', '用户态与内核态边界；普通 read/write 是同步调用，不等于同步持久化。'),
  cache: role('cache', '客户端 page cache', '保存文件内容；clean / dirty / writeback 与是否驻留是不同维度。'),
  queue: role('queue', '块层 / 请求队列', '本地块映射、排队、合并和拆分请求的示意；一次 API 不等于一次设备 I/O。'),
  device: role('device', '驱动 / 设备', '可能命中控制器或设备缓存；设备到内存的 DMA 不消除内核到用户态的复制。'),
  server: role('server', '网络 / 服务端', '客户端 RPC 到服务端文件系统或对象层；服务端与设备仍可能命中缓存。'),
  stable: role('stable', '持久化承诺', '表示应用已获得的同步保证，不推断 NAND/盘片的每个实际写入时点。'),
  python: role('python', 'Python 文件对象缓冲', '只适用于内置 open() 的 buffered 文件对象；os.write 没有这一层。'),
  clock: role('clock', 'perf_counter 计时', '展示代码边界，不画假造的毫秒数、带宽或耗时比例。'),
}
const event = (from, to, payload, kind = '控制 / 请求') => ({ from, to, payload, kind })
const page = (label, value, status) => ({ label, value, status })
const initialState = () => ({
  app: '尚未调用', kernel: '尚未处理', cache: '未观察', backend: '尚未访问',
  durability: '本组不建立新的持久化保证', offset: '—', user: '尚无返回数据',
  metadata: '文件已有；inode #100（示意）', pages: [], handles: [],
  total: '未开始', latency: '未开始',
})
function builder() {
  let state = initialState()
  const steps = []
  return {
    steps,
    add(title, description, code, patch = {}, events = [], note = '') {
      state = { ...state, ...patch }
      steps.push({ title, description, code, state: structuredClone(state), events, note })
    },
  }
}

export const IO_DEMOS = {
  open: {
    id: 'open', label: '打开文件', title: '打开文件：名字、inode、打开对象和 fd',
    goal: '看清两个 fd 究竟共享 inode、偏移，还是仅共享内容缓存。',
    variants: [{ id: 'independent', label: '再次独立 open' }, { id: 'dup', label: 'dup 共享打开对象' }],
    assumptions: ['已存在的普通文件、检查成功；未展示符号链接、创建失败和安全模块的所有细节。', '目录权限检查分布在路径查找中，图中集中说明，不是固定的内核函数调用顺序。'],
  },
  read: {
    id: 'read', label: '读取文件', title: '读取：命中、冷读与预读',
    goal: '跟踪文件范围、缓存页、后端请求、用户缓冲区和当前偏移。',
    variants: [{ id: 'hit', label: 'page cache 命中' }, { id: 'miss', label: 'page cache 未命中' }, { id: 'readahead', label: '顺序读 + 预读' }],
    assumptions: ['每块用 4 KiB 表示；脚本顺序读实际用 1 MiB 缓冲区。假设本次没有短读和错误，文件不是稀疏洞。', '预读窗口仅为教学示例，不是 Linux 固定窗口；未画出的其他线程也可能改变缓存。', 'readv / preadv 只描述用户缓冲区与定位方式，不自动代表 direct、异步、零复制或物理介质读取。'],
  },
  write: {
    id: 'write', label: '写入文件', title: '写入：dirty、回写与同步边界',
    goal: '看到 write 已返回而后端持久化尚无保证的合法状态。',
    variants: [{ id: 'batch', label: '两块写完再 fsync' }, { id: 'each', label: '每块写完都 fsync' }, { id: 'pressure', label: '脏页压力使 write 阻塞' }],
    assumptions: ['用两次完整 4 KiB 覆盖写表示更大负载；覆盖前 A0/A1 已同步，新内容是 B0/B1，没有并发修改者。', '上次同步记录是历史，不是覆盖写之后的可恢复旧版本保证。未同步的新内容可能已部分回写，崩溃后不保证完整旧版本或新版本。', '回写、设备缓存和 fsync 的具体实现依文件系统而异；省略日志、复制、写放大及错误处理。'],
  },
  sync: {
    id: 'sync', label: '同步与驱逐', title: 'flush、fsync、DONTNEED、close 各做什么',
    goal: '把用户态提交、持久化保证、缓存驻留和句柄生命周期分开。',
    variants: [{ id: 'evicted', label: '同步后建议驱逐成功' }, { id: 'retained', label: '同步后页面仍驻留' }, { id: 'dirty', label: '未同步就建议驱逐' }, { id: 'close', label: '只 flush + close' }],
    assumptions: ['Python 缓冲路径假设小写入尚未自行触发 flush；脚本使用 os.write，不经过这层文件对象缓冲。', '这里只演示文件内容，不是完整的临时文件、原子 rename、目录 fsync 崩溃安全协议。', 'clean 页面也可能仍驻留；DONTNEED 是建议，既不清理所有层级缓存，也不能替代同步。'],
  },
  benchmark: {
    id: 'benchmark', label: '压测计时', title: '六个 phase：逐行看 API 和计时边界',
    goal: '区分总吞吐计时、单次 latency、setup 和 finally 中的 close。',
    variants: [
      { id: 'seq_write_buf', label: 'seq_write_buf' }, { id: 'seq_write_sync', label: 'seq_write_sync' },
      { id: 'seq_read', label: 'seq_read' }, { id: 'rand_write_4k_buf', label: 'rand_write_4k_buf' },
      { id: 'rand_read_4k', label: 'rand_read_4k' }, { id: 'rand_write_4k_sync', label: 'rand_write_4k_sync' },
    ],
    assumptions: ['按 tools/disk_io_bench.py 的实际控制流抽取；重复循环只展示一个代表性操作，不虚构性能数字。', '写函数在 close 后计算返回耗时；读函数先求 return 表达式，再执行 finally 中的 close。', '应用一次一个未返回调用，不代表块设备始终只有一个在途请求。'],
  },
}

function buildOpen(variant) {
  const b = builder()
  b.add('提供路径', '应用提供文件名，没有 fd；打开文件不会自动加载全部内容。', 'os.open("/data/checkpoint.bin", os.O_RDONLY)',
    { app: '提出只读打开请求', user: '没有读取内容', metadata: '路径尚待解析' }, [event('app', 'path', '/data/checkpoint.bin')])
  b.add('逐级解析与缓存查找', '沿目录分量查 dentry / inode 缓存；未命中时可能需要本地元数据 I/O 或服务端查找。', 'lookup("data") → lookup("checkpoint.bin")',
    { kernel: '解析路径 / 挂载点', metadata: '目录名字与对象关系查找中' }, [event('path', 'dentry', '父目录 + 名字', '元数据')])
  b.add('找到文件对象并检查访问', '名字关联到 inode #100，检查沿途目录和目标文件权限及打开模式。目录记录和内存 dentry 不是同一对象。', 'dentry → inode #100; check access',
    { kernel: '本例检查通过', metadata: 'inode #100；大小 16 KiB；已有普通文件' }, [event('dentry', 'inode', '名字 → inode #100', '元数据')])
  b.add('建立打开对象 A', '创建这一次打开的访问关系，记录 offset 和 flags；当前偏移不属于 inode 的全局游标。', 'open-file A: offset = 0',
    { kernel: '打开文件对象 A 已创建；尚未向进程交付 fd', offset: 'A = 0', handles: [] }, [event('inode', 'file', '对象 A 关联 inode #100', '元数据')])
  b.add('返回 fd 3', '进程 fd 表的整数索引 3 关联打开对象 A；此刻仍没有读取文件内容。', 'fd3 = 3',
    { app: 'open 已返回 fd 3', kernel: 'fd 表已登记', user: '没有读取内容', handles: [{ fd: 3, object: 'A', offset: 0, inode: 100 }] }, [event('file', 'fd', 'fd 3 → 打开对象 A', '返回')])
  const duplicate = variant === 'dup'
  b.add(duplicate ? 'dup：另一个 fd，同一个打开对象' : '独立 open：同一 inode，新的打开对象',
    duplicate ? 'dup(fd3) 创建 fd 4，但仍指向打开对象 A，所以共享偏移。不是再次解析完整路径。' : '再次独立 open 同一路径获得 fd 4 和打开对象 B。B 与 A 关联同一 inode，但各有自己的偏移。',
    duplicate ? 'fd4 = os.dup(fd3)' : 'fd4 = os.open(path, os.O_RDONLY)',
    { app: '拿到 fd 3 和 fd 4', offset: duplicate ? 'A = 0（fd 3/4 共享）' : 'A = 0；B = 0', handles: [{ fd: 3, object: 'A', offset: 0, inode: 100 }, { fd: 4, object: duplicate ? 'A' : 'B', offset: 0, inode: 100 }] },
    [event(duplicate ? 'file' : 'inode', 'fd', duplicate ? 'fd 4 → A' : 'fd 4 → B → 同一 inode', '元数据')])
  b.add('通过 fd 3 读入 4 KiB', '本例完整读取 4096 字节，推进 fd 3 对应打开对象的偏移；对照 fd 4 是否一起变化。', 'data = os.read(fd3, 4096)',
    { app: 'read 返回 4096 字节', user: '4096 字节', offset: duplicate ? 'A = 4096（两者都能观察到）' : 'A = 4096；B = 0', handles: [{ fd: 3, object: 'A', offset: 4096, inode: 100 }, { fd: 4, object: duplicate ? 'A' : 'B', offset: duplicate ? 4096 : 0, inode: 100 }] },
    [event('file', 'app', '返回 4096 字节；更新对应打开对象偏移', '返回')], '是否命中内容缓存与是否共享偏移是两件事；本组不展开这次读取的数据路径。')
  return { steps: b.steps, nodes: ['app', 'path', 'dentry', 'inode', 'file', 'fd'] }
}

function buildRead(variant, api, backend) {
  const b = builder()
  const sequential = variant === 'readahead'
  const positional = !sequential && api === 'preadv'
  const name = positional ? 'preadv' : 'readv'
  const target = positional ? 8192 : 4096
  const value = positional ? 'C' : 'B'
  const label = `[${target}, ${target + 4096})`
  const network = backend === 'network'
  const endpoint = network ? 'server' : 'device'
  const pages = [page(label, value, variant === 'hit' ? 'clean' : 'missing')]
  const filled = [page(label, value, 'clean')]
  const preloaded = [...filled, page('[8192, 12288)', 'C', 'prefetched')]
  b.add('准备请求与用户缓冲', '文件已有 A/B/C/D 四块，每块 4 KiB；打开对象当前偏移为 4096。切换定位 API 可观察实际目标范围变化。',
    positional ? 'buf = bytearray(4096); offset = 8192' : 'buf = bytearray(4096); current_offset = 4096',
    { app: '准备 4 KiB 读取', offset: '当前偏移 = 4096', user: '0 / 4096 字节有效', pages, cache: variant === 'hit' ? '目标页已缓存' : '目标页未缓存', backend: '本次尚未提交后端数据请求' })
  b.add(`进入 ${name}`, '内核根据 fd 找到打开对象、访问模式和关联 inode；本次调用尚未返回。',
    positional ? 'n = os.preadv(fd, [buf], 8192)' : 'n = os.readv(fd, [buf])',
    { app: '调用中，等待返回', kernel: `处理 ${name}，目标 ${label}` }, [event('app', 'syscall', `${name}：4 KiB，目标 ${label}`)])
  b.add('确认文件身份与逻辑范围', '通过已打开关系确认 inode 和文件范围；逻辑 offset 不是物理块地址。普通 buffered 读先查内容缓存。', 'fd → 打开对象 → inode #100 → page cache',
    { metadata: `inode #100；目标 ${label}`, kernel: '检查文件范围并查询目标缓存页' }, [event('syscall', 'inode', `inode #100；逻辑范围 ${label}`, '元数据'), event('inode', 'cache', `查询文件页 ${label}`, '缓存查询')])
  if (variant === 'hit') {
    b.add('page cache 命中：后端数据读跳过', '所需内容已在客户端内核缓存中，无须为此次读取向块设备或服务端请求数据。', 'cache hit → skip backend data read',
      { cache: `命中 clean 页 ${value}`, backend: '本次没有后端数据读', kernel: '数据已可用，准备复制' }, [], '元数据一致性检查仍可能存在；这里跳过的是后端文件内容读取，不宣称所有网络活动都为零。')
  } else {
    b.add('page cache 未命中', '客户端缺少这段内容；接下来才需要做具体布局映射或发起网络文件请求。', 'cache miss → resolve layout / backend request',
      { cache: '目标页缺失', kernel: '准备请求后端', backend: '尚未提交' }, [event('cache', 'inode', '未命中，准备定位后端范围', '元数据')])
    if (!network) {
      b.add(sequential ? '组织读取与预读窗口' : '映射文件块并排队', sequential ? '本例请求 B，并假设内核顺便预读下一块 C。实际预读窗口不固定，应用尚未请求 C。' : '文件系统查 extent 等结构，把逻辑范围映射为块请求；块层可能合并、拆分并排队。',
        sequential ? 'submit read(B) + illustrative readahead(C)' : 'file offset → extent → block request',
        { kernel: '提交本地块请求', backend: sequential ? '读取 B，另预取 C（示意）' : `目标 ${label} 请求排队` }, [event('inode', 'queue', sequential ? '请求 B + 预读 C' : `读取 ${label}`, '后端请求')])
      b.add('驱动与设备执行', '设备可能从自身缓存满足请求；本例选择设备到内核内存的 DMA 路径，线程等待数据可用。未指定一定读取 NAND/盘片。', 'queue → driver → device/cache → memory',
        { app: '等待数据', backend: '请求执行中；介质/缓存来源未区分' }, [event('queue', 'device', '提交块请求', '后端请求')])
    } else {
      b.add('经网络请求服务端', 'NFS/CephFS 等客户端向相应服务端文件系统或对象节点请求；图中将多个网络角色合并为一个服务端盒子。', 'client request → network → server/filesystem',
        { app: '等待网络与服务端完成', backend: sequential ? '服务端处理目标与预读范围' : '服务端处理目标范围' }, [event('inode', 'server', sequential ? '读取 B，示例中另取 C' : `读取 ${label}`, '网络请求')])
      b.add('服务端取得数据', '客户端未命中，服务端仍可能从服务端 page cache、对象缓存或设备缓存返回内容，不能据此认定物理冷读。', 'server cache / filesystem / device',
        { backend: '服务端内容已可用，数据源仍可能是缓存' }, [event('server', 'server', '服务端内部取数；省略其内部层级', '服务端处理')])
    }
    b.add(sequential ? '目标页与预读页填入客户端缓存' : '填充客户端 page cache', '数据先在内核路径上变得可用；用户缓冲区此刻还没有因为 DMA 就自动得到这些字节。', 'backend data → client page cache',
      { cache: sequential ? 'B 已可读，C 在本例已预读完成' : `目标页 ${value} 已可读`, pages: sequential ? preloaded : filled, backend: '本例请求完成；不推断持久化', user: '仍为 0 / 4096 字节有效' },
      [event(endpoint, 'cache', sequential ? 'B + 示例预读 C → 缓存' : `${value} → 内核缓存`, network ? '网络返回' : 'DMA / 内核内存')])
  }
  b.add('内核复制到用户缓冲区', '普通 buffered read 把内核缓存中的请求范围复制到应用 buf；这一步与设备到内存的 DMA 是不同的数据传输。', 'copy page-cache data → buf[0:4096]',
    { user: `4096 / 4096 字节有效，内容 ${value}`, kernel: '完成用户缓冲区填充' }, [event('cache', 'app', `${value}：4096 字节 → buf`, '数据复制')])
  b.add('系统调用返回', positional ? 'preadv 返回 4096，显式 offset 是 8192，但打开对象当前偏移仍为 4096。' : 'readv 返回 4096，当前偏移从 4096 推进到 8192。短读/EOF/错误需要另行处理，本例没有发生。',
    'return n = 4096', { app: `${name} 已返回`, kernel: '本次调用结束', offset: positional ? '当前偏移 = 4096（不变）' : '当前偏移 = 8192' }, [event('syscall', 'app', 'n = 4096', '返回')])
  if (sequential) {
    b.add('下一次 readv 请求 C：命中本轮预读', '应用现在才请求 C，但 C 已在前一个预读窗口进入缓存。冷启动顺序读的后续调用也可能命中本轮缓存。', 'next readv → cache hit(C) → return 4096',
      { app: '下一次 readv 也已返回', offset: '当前偏移 = 12288', user: '复用 buf：现为 C 的 4096 字节', backend: '这次访问 C 没有新增后端数据请求' }, [event('cache', 'app', '预读页 C → 复用的 buf', '数据复制')])
  }
  return { steps: b.steps, nodes: ['app', 'syscall', 'inode', 'cache', ...(network ? ['server'] : ['queue', 'device'])] }
}

function buildWrite(variant) {
  const b = builder()
  const each = variant === 'each'
  const pressure = variant === 'pressure'
  let pages = [page('块 0', '—', 'missing'), page('块 1', '—', 'missing')]
  const guarantee = ['A0', 'A1']
  const pending = new Set()
  const guaranteeText = () => `上次同步记录：${guarantee.join(' / ')}；${pending.size ? `覆盖后尚未确认同步：块 ${[...pending].join('、')}；崩溃后不保证完整旧/新版本` : '本例当前版本均已同步（无并发修改）'}`
  const sync = (indices, label) => {
    b.add(`调用 fsync：${label}`, '应用进入 fsync 并等待文件数据和相关元数据满足文件系统同步承诺。此时还不能把等待中的调用说成成功。', 'os.fsync(fd)  # 尚未返回',
      { app: '等待 fsync', kernel: '推进文件同步', durability: guaranteeText() }, [event('app', 'syscall', 'fsync 请求', '同步请求')])
    pages = pages.map((p, i) => indices.includes(i) ? { ...p, status: 'writeback' } : p)
    b.add('回写并推进后端稳定边界', '图中把数据回写、必要元数据/事务以及后端稳定存储要求合并为一步；不同系统可能使用 flush、FUA、网络稳定写等不同机制。', 'writeback + required metadata/backend synchronization',
      { cache: '本例相关新页正在回写', pages, backend: '等待数据及必要同步要求完成' }, [event('cache', 'device', '新数据 / 必要文件状态', '回写'), event('syscall', 'stable', '推进并等待稳定边界', '同步请求')])
    pages = pages.map((p, i) => indices.includes(i) ? { ...p, status: 'clean' } : p)
    for (const i of indices) { guarantee[i] = `B${i}`; pending.delete(i) }
    b.add('fsync 成功返回', '没有并发修改者时，本例同步页为 clean，但仍驻留缓存。父目录项不因此自动获得单独同步保证。', 'fsync returns success',
      { app: 'fsync 已成功返回', kernel: '本次同步结束', cache: '同步页 clean，仍可驻留', pages, backend: '报告本例文件同步完成', durability: guaranteeText() }, [event('stable', 'app', '文件同步成功', '完成确认')])
  }
  b.add('已有 A0/A1，准备 B0/B1', '两个 4 KiB 范围已有旧内容 A0/A1。应用准备新数据，当前还没有写入 page cache。', 'payload0 = B0; payload1 = B1',
    { app: '准备两次覆盖写', pages, cache: '目标页尚未缓存', user: 'B0 / B1 在用户缓冲区', offset: 'pwrite 目标 = 0、4096；当前偏移不变', durability: guaranteeText(), backend: '初始稳定内容 A0 / A1' })
  for (let i = 0; i < 2; i++) {
    b.add(`进入 pwrite：块 ${i}`, '显式逻辑 offset 不改变打开对象当前偏移；本例完整写满 4 KiB，短写时真实脚本会继续重试。', `os.pwrite(fd, B${i}, ${i * 4096})`,
      { app: `pwrite ${i} 调用中`, kernel: `准备接受 B${i}` }, [event('app', 'syscall', `B${i}，offset ${i * 4096}`)])
    if (pressure && i === 1) {
      b.add('脏页压力：第二次 write 暂时阻塞', '这里主动选择回写限速路径：系统已有脏页压力，内核让写线程等待回写进展。不是说所有 8 KiB 写入都会触发限速。', 'write waits for dirty-page throttling',
        { app: '第二次 pwrite 尚未返回', kernel: '回写限速，等待进展', cache: 'B0 dirty；B1 尚未被接受' }, [event('syscall', 'cache', '等待脏页预算 / 回写进展', '等待')])
      pages = pages.map((p, j) => j === 0 ? { ...p, status: 'writeback' } : p)
      b.add('后台回写为写线程腾出空间',  '后台推动 B0 回写，使第二个调用随后可以继续。本例没有在这里向应用返回 fsync 成功，因此不把历史同步记录更新为 B0；也不保证旧 A0 完整保留。', 'background writeback → writer may resume',
        { app: '仍在第二次 pwrite 内', cache: 'B0 回写中', pages, backend: '后台请求在途' }, [event('cache', 'device', 'B0 后台回写', '回写')])
    }
    pages = pages.map((p, j) => j === i ? { ...p, value: `B${i}`, status: 'dirty' } : p)
    pending.add(i)
    b.add(`B${i} 进入 page cache，标记 dirty`, '用户数据复制到内核页；文件大小、时间和布局等元数据也可能变化，本例为已有文件覆盖写，不扩展大小。', `copy B${i} → page cache; mark dirty`,
      { cache: `块 ${i} = B${i} dirty`, pages, kernel: '已接受本例完整 4 KiB', metadata: 'inode #100；覆盖写，mtime/ctime 等可能变化', durability: guaranteeText(), backend: '新数据可被后台写入；当前介质内容未观测' }, [event('syscall', 'cache', `B${i}：4096 字节`, '数据复制')])
    b.add(`pwrite ${i} 返回，但不承诺新数据已持久化`, '内核接受数据与同步持久化是两个完成层次；上次同步记录只是历史。原地覆盖后发生崩溃，既不保证完整新版本，也不保证恢复完整旧版本。', 'return n = 4096',
      { app: `pwrite ${i} 已返回 4096`, kernel: '本次写调用返回', durability: guaranteeText() }, [event('syscall', 'app', 'n = 4096，不含持久化保证', '返回')])
    if (each) sync([i], `块 ${i}`)
  }
  if (!each) sync([0, 1], '两块一起')
  b.add('对照终态', each ? '本例对每块分别同步，共两个主要同步边界；这不是脚本末尾额外 fsync 的完整计时模型，精确脚本控制流见压测计时演示。' : '两次应用写入之后统一同步，给批处理和后端合并留下空间；实际是否合并、加速多少必须测量。', 'close(fd)  # 释放访问关系，不驱逐 clean 页',
    { app: '演示结束，释放 fd', kernel: '不再持有本例访问句柄', cache: 'B0 / B1 clean；可能仍驻留', durability: guaranteeText() }, [event('app', 'fd', 'close：释放句柄', '生命周期')])
  return { steps: b.steps, nodes: ['app', 'syscall', 'inode', 'cache', 'device', 'stable', 'fd'] }
}

function buildSync(variant) {
  const b = builder()
  const dirtyOnly = variant === 'dirty'
  const closeOnly = variant === 'close'
  b.add('已有稳定旧内容 A', '本例没有并发写入者。A 已稳定，新内容 B 还在应用侧；页面是否驻留与内容是否持久化分别记录。', 'initial: stable A; new payload B',
    { user: 'B 在用户内存', durability: '已确认稳定版本 A', backend: '旧内容 A', pages: [page('目标页', '—', 'missing')] })
  if (dirtyOnly) {
    b.add('os.write 直接接受 B', '脚本使用 os.write，没有 Python 文件对象缓冲；内核接受数据后页变 dirty，本例暂不调用 fsync。', 'os.write(fd, B)',
      { app: 'write 已返回', cache: 'B dirty', pages: [page('目标页', 'B', 'dirty')], kernel: '已接受数据，尚未建立同步边界', durability: '上次同步记录 A；覆盖后的 B 无新增同步保证；也不保证恢复完整 A', backend: '原地覆盖可能已开始后台写入，当前介质内容未观测' }, [event('app', 'cache', 'B → page cache', '数据复制')])
    b.add('对 dirty 页发 DONTNEED 建议', 'DONTNEED 不能把尚未安全处理的脏数据任意丢掉。本例选择脏页保留的合法结果；该调用不替应用建立 fsync 保证。', 'os.posix_fadvise(fd, 0, 0, DONTNEED)',
      { app: '建议调用已返回', cache: '本例 B dirty 仍保留', durability: 'B 无新增同步保证；上次同步 A 是历史，不保证崩溃后仍为完整 A' }, [event('app', 'cache', 'DONTNEED：建议，不是命令', '缓存建议')])
    b.add('应先同步，再另行请求回收', '脚本先完整写入并 fsync，再建议回收 clean 页。驱逐和持久化是两件事，不能任选其一。', 'fsync first; then optionally DONTNEED',
      { app: '当前分支到此结束，尚未执行修复性 fsync', kernel: '本例没有新的同步成功确认' }, [], '这里只解释正确顺序，不用一句建议文本假装已经执行了 fsync。')
  } else {
    b.add('f.write：先留在 Python 缓冲', '选择小写入仍留在 Python 文件对象缓冲的路径；不表示每个 f.write 都一定不会进入内核。', 'f.write(B)',
      { app: '文件对象接受 B', kernel: '本例尚未收到 B', user: 'Python 文件对象缓冲含 B' }, [event('app', 'python', 'B 在用户态缓冲', '数据复制')])
    b.add('f.flush：交给操作系统', '把 Python 文件对象缓冲提交到内核；数据成为 page cache 中的 dirty 页，但还没有新的持久化保证。', 'f.flush()',
      { app: 'flush 完成', kernel: '已接受 B', user: 'Python 缓冲已提交', cache: 'B dirty', pages: [page('目标页', 'B', 'dirty')], durability: '上次同步记录 A；覆盖后的 B 未同步，也不保证完整旧 A 可恢复', backend: 'B 可能已开始后台写入；未观测介质内容' }, [event('python', 'cache', 'B → 内核；没有稳定承诺', '数据复制')])
    if (closeOnly) {
      b.add('只 close：释放句柄', '文件对象正常关闭会处理其缓冲并释放 fd，但不是通用的 fsync 持久化屏障。后端可能已在后台写入，应用没有获得显式同步确认。', 'f.close()  # no explicit fsync',
        { app: '文件对象已关闭', kernel: 'fd 已释放；后台回写仍可进行', durability: 'B 未获得显式 fsync 保证' }, [event('app', 'fd', 'close / 释放 fd', '生命周期')])
    } else {
      b.add('fsync：进入等待', '请求文件内容和相关元数据达到文件系统同步边界；同步不代表所有缓存都被清空。', 'os.fsync(f.fileno())  # 等待中',
        { app: '等待 fsync', kernel: '推进回写与必要同步', cache: 'B 回写中', pages: [page('目标页', 'B', 'writeback')], backend: 'B 同步处理中' }, [event('cache', 'device', 'B 回写', '回写'), event('app', 'stable', 'fsync 请求', '同步请求')])
      b.add('fsync 成功：clean，但仍驻留', '本例没有并发修改者，B 已满足文件系统同步承诺；内核仍可保留 clean 页供之后读取。', 'fsync returns success',
        { app: 'fsync 已返回', kernel: '同步完成', cache: 'B clean 且仍驻留', pages: [page('目标页', 'B', 'clean')], durability: '已确认文件内容 B 的同步承诺', backend: '已报告本例同步完成' }, [event('stable', 'app', '同步成功；不是驱逐完成', '完成确认')])
      const retained = variant === 'retained'
      b.add('DONTNEED：回收建议的两种结果', retained ? '选择“页面仍保留”的合法分支；即便同步成功，该建议也不承诺一定驱逐。' : '选择“本次目标 clean 页已回收”的合法分支。这个结果不能由建议调用成功返回普遍推导。', 'os.posix_fadvise(fd, 0, 0, DONTNEED)',
        { app: '建议调用返回', cache: retained ? 'B clean 仍驻留（本例结果）' : '目标页已回收（本例结果）', pages: [page('目标页', retained ? 'B' : '—', retained ? 'clean' : 'missing')], backend: '服务端、控制器和设备缓存未被该建议清理' }, [event('app', 'cache', '建议回收客户端文件页', '缓存建议')])
      b.add(retained ? '随后读取：客户端可直接命中' : '随后读取：仍可能命中后端缓存', retained ? '本例目标页仍在客户端缓存，读取可以直接复制；先前 fsync 并没有要求把它删掉。' : '客户端目标页已回收，需要向后端取数据，但服务端或设备缓存仍可能命中，不保证读物理介质。', 'read after advice',
        { app: '后续读取已取得 B', user: '用户缓冲区得到 B', cache: 'B 再次/继续驻留', pages: [page('目标页', 'B', 'clean')], backend: retained ? '本次无后端数据读' : '后端返回 B，可能命中后端缓存' },
        retained ? [event('cache', 'app', 'B → buf', '数据复制')] : [event('device', 'cache', 'B，来源可能是下层缓存', '后端返回'), event('cache', 'app', 'B → buf', '数据复制')])
    }
  }
  return { steps: b.steps, nodes: ['app', ...(dirtyOnly ? [] : ['python']), 'cache', 'device', 'stable', 'fd'] }
}

function buildBenchmark(variant) {
  const b = builder()
  const isRead = variant === 'seq_read' || variant === 'rand_read_4k'
  const isRandom = variant.startsWith('rand_')
  const syncEach = variant.endsWith('_sync')
  const op = isRead ? (isRandom ? 'os.preadv(fd, [buf], off)' : 'os.readv(fd, [buf])') : (isRandom ? 'os.pwrite(fd, view, off + completed)' : 'os.write(fd, view)')
  const scriptFunction = isRead ? (isRandom ? 'bench_rand_read' : 'bench_seq_read') : (isRandom ? 'bench_rand_write' : 'bench_seq_write')
  b.add('setup：open，不计入 phase 总时间', '传入参数、offset/payload 生成和 scratch 目录创建等更外层工作也不计入该 phase。这里展示函数内的打开。', 'fd = os.open(path, flags)',
    { app: `${variant}：进入 ${scriptFunction}`, kernel: '打开文件', total: '未开始', latency: isRandom ? '未开始' : '不输出逐操作样本' }, [event('app', 'syscall', 'open：setup', '未计时')])
  if (isRead) {
    b.add('驱逐建议与缓冲分配：不计时', 'DONTNEED 和 bytearray 分配发生在 t0 之前；驱逐只是建议，不保证物理冷读。', 'posix_fadvise(DONTNEED); buf = bytearray(...)',
      { cache: '发出回收建议；具体结果不保证', total: '未开始' }, [event('app', 'cache', 'DONTNEED + 用户缓冲准备', '未计时')])
  } else if (!isRandom) {
    b.add('准备 payload：不计时', '顺序写在 t0 之前生成 os.urandom(block)。这不是整个文件的预写入，而是生成后续复用的用户态块。', 'payload = os.urandom(block)',
      { user: '1 MiB payload 已准备', total: '未开始' })
  }
  b.add('开始总计时', 'perf_counter 记录 t0；之后的循环和计时内同步成本都会进入吞吐分母。', 't0 = time.perf_counter()',
    { total: '计时中', app: '进入操作循环' }, [event('app', 'clock', '总计时开始', '计时边界')])
  if (isRandom) {
    b.add('开始当前操作 latency', '每个随机操作在其写满/读满循环前记录 s；offset 序列已经在函数调用前生成。', 's = time.perf_counter()',
      { latency: '当前样本计时中', total: '计时中' }, [event('app', 'clock', '当前操作 latency 开始', '计时边界')])
  }
  b.add(isRead ? '执行读取 / 读满循环' : '执行写入 / 写满循环', isRead ? '顺序读请求 1 MiB；随机读请求 4 KiB，并处理短读直到读满，随机读意外 EOF 报错。' : '顺序写每块最大 1 MiB；随机写为 4 KiB；短写要继续完成剩余字节。write 返回不等于持久化。', op,
    { app: '操作调用处理中', kernel: op, total: '计时中', cache: isRead ? '可能命中、后端读取或预读' : '新数据可能进入 dirty page' }, [event('app', 'syscall', isRead ? 'readv / preadv' : 'write / pwrite', '总计时内')])
  if (!isRead && syncEach) {
    b.add('每块 / 每次操作 fsync', isRandom ? '当前随机写样本尚未结束；pwrite 和这个 fsync 都包含在同一 latency 内。' : '每写满一个 1 MiB 块就 fsync；此步骤仍走 buffered I/O，不是 O_DIRECT。', 'os.fsync(fd)',
      { app: '等待逐操作同步', total: '计时中', latency: isRandom ? '当前样本仍在计时' : '不输出逐操作样本' }, [event('syscall', 'stable', '逐操作 fsync', '总计时内')])
  }
  if (isRandom) {
    b.add('记录当前 latency', isRead ? '记录一次完整 4 KiB 读取的 latency，后续输出平均值。' : syncEach ? 'fsync_lat_p99 的样本实际是完整 pwrite + fsync。' : '这个样本只含写满循环，不含循环后的最终 fsync；脚本不把它作为落盘延迟输出。', 'lat.append(time.perf_counter() - s)',
      { latency: '当前样本已捕获；下一次操作会重新开始', total: '计时中', app: '继续后续随机操作' }, [event('clock', 'app', '本次 latency 捕获', '计时边界')])
  }
  b.add('完成其余循环', isRead && !isRandom ? '顺序读持续到 readv 返回 0（EOF），最后这次 EOF 调用也在总计时内。' : '继续相同模式的其余操作；这里不为每次重复绘制相同步骤。循环、Python 记账和时钟采样开销仍在总计时内。', isRead && !isRandom ? 'while ...: readv(...); n == 0 → break' : 'repeat remaining operations',
    { app: '操作循环结束', total: '计时中', latency: isRandom ? '全部样本已捕获' : '不输出逐操作样本' })
  if (!isRead && (!syncEach || !isRandom)) {
    b.add(syncEach ? '顺序 sync 的末尾额外 fsync' : '循环后的最终 fsync', syncEach ? 'bench_seq_write 无条件在循环后再次 fsync，因此 seq_write_sync 有每块同步和最后额外同步。' : '最终 fsync 在总吞吐计时内；buffered 写阶段不是只测 page cache 接收速度。随机写的逐操作 latency 不包含这次统一同步。', 'os.fsync(fd)  # final',
      { total: '计时中', app: '等待最终同步' }, [event('syscall', 'stable', '最终 fsync', '总计时内')])
  }
  if (isRead) {
    b.add('先捕获 return 表达式中的总耗时', 'Python 先求 return 表达式，其中 perf_counter()-t0 已算好；然后才执行 finally。不是 close 之后重新计算。', 'return total_or_ops, time.perf_counter() - t0, ...',
      { total: '总耗时已捕获（不再累计）', app: 'return 值已求出，尚待 finally' }, [event('clock', 'app', '捕获 dt，结束计时', '计时边界')])
    b.add('finally：close 不在已捕获 dt 中', '函数仍会正常关闭 fd；只是这次 close 的成本没有进入此前已经求好的读阶段 dt。', 'finally: os.close(fd)',
      { total: '已捕获；close 不计入', app: '关闭完成后交付返回值', kernel: '释放 fd' }, [event('app', 'fd', 'close：读阶段计时外', '未计时')])
  } else {
    b.add('finally：close 仍在总计时内', '写函数 return 语句位于 finally 之后，所以关闭 fd 时总计时还没有捕获终点；单次 latency 不包含 close。', 'finally: os.close(fd)',
      { total: '计时中；close 也计入', app: '关闭 fd', latency: isRandom ? '全部样本已结束；不含 close' : '不输出逐操作样本' }, [event('app', 'fd', 'close：写阶段总计时内', '总计时内')])
    b.add('close 后捕获总耗时', '此处才计算 perf_counter()-t0；结果包含循环、必要 fsync 和 close，但不包含前面的 open/setup。', 'return count, time.perf_counter() - t0, ...',
      { total: '总耗时已捕获（包含 close）', app: '返回本 phase 结果' }, [event('clock', 'app', '捕获 dt，结束计时', '计时边界')])
  }
  return { steps: b.steps, nodes: ['app', 'syscall', 'cache', 'stable', 'clock', 'fd'], scriptFunction }
}

export function getIoScenario(mode = 'open', variant, options = {}) {
  const base = IO_DEMOS[mode] || IO_DEMOS.open
  const selected = base.variants.some(item => item.id === variant) ? variant : base.variants[0].id
  let result
  if (base.id === 'open') result = buildOpen(selected)
  else if (base.id === 'read') result = buildRead(selected, options.api, options.backend)
  else if (base.id === 'write') result = buildWrite(selected)
  else if (base.id === 'sync') result = buildSync(selected)
  else result = buildBenchmark(selected)
  return {
    ...base, ...result, variant: selected,
    nodes: result.nodes.map(id => ROLES[id]),
    assumptions: [...base.assumptions, '动画步长没有实际耗时含义；未把示意状态当作真实磁盘观测数据。'],
  }
}

export const STATE_FIELDS = [
  ['app', '应用 / 调用'], ['kernel', '内核处理'], ['cache', '客户端内容缓存'],
  ['backend', '后端路径'], ['durability', '应用已获持久化保证'], ['offset', '文件位置'],
  ['user', '用户数据'], ['metadata', '文件元数据'],
]
export const PAGE_LABELS = { missing: '未驻留', clean: 'clean / 可读', dirty: 'dirty / 待回写', writeback: '回写中', prefetched: '预读完成 / clean' }
