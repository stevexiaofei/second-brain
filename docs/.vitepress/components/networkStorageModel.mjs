// Each step is a self-contained teaching state, not a trace of every protocol message.
export const STORAGE_SCENARIOS = {
  nfs: {
    title: 'NFS：从挂载到读取',
    assumptions: [
      '服务器将 /srv/nfs/datasets 作为 fsid=0 的根导出；客户端以 10.20.0.10:/ 挂载到 /mnt/datasets。',
      'public 指客户端可达的存储网络，不等于互联网；服务器可能从缓存或文件系统取得数据，不保证每次访问磁盘。',
    ],
    nodes: [
      { id: 'gpu-c1', label: 'gpu-c1', detail: 'NFS 客户端' },
      { id: 'public-switch', label: 'public-switch', detail: 'public 网络交换机' },
      { id: 'nfs-server', label: 'nfs-server', detail: '10.20.0.10' },
      { id: 'disk', label: '缓存 / 文件系统', detail: '服务端数据来源，不一定读盘' },
    ],
    steps: [
      { title: '尚未挂载', description: 'gpu-c1 上的 /mnt/datasets 只是本地目录；访问它不会访问远端导出。', activeNodes: ['gpu-c1'], messages: [] },
      { title: '挂载远端导出', description: 'VFS 将本地挂载点映射到远端根导出，而不是下载或复制整棵目录。', activeNodes: ['gpu-c1', 'public-switch', 'nfs-server'], messages: [{ from: 'gpu-c1', to: 'nfs-server', network: 'public（经 public-switch）', payload: '挂载 10.20.0.10:/ → /mnt/datasets' }] },
      { title: '打开并读取文件', description: '应用打开 /mnt/datasets/train/image001.jpg 并调用 read；VFS 将访问交给 NFS 客户端。', activeNodes: ['gpu-c1'], messages: [{ from: 'gpu-c1', to: 'gpu-c1', network: '本地', payload: 'open / read：/mnt/datasets/train/image001.jpg' }] },
      { title: '发送读取请求', description: '客户端通过 public 网络向服务器请求文件数据；交换机只转发网络流量。', activeNodes: ['gpu-c1', 'public-switch', 'nfs-server'], messages: [{ from: 'gpu-c1', to: 'nfs-server', network: 'public（经 public-switch）', payload: '读取远端 train/image001.jpg' }] },
      { title: '服务器取得数据', description: '服务器从缓存或文件系统取得文件数据；缓存命中时不一定读取磁盘。', activeNodes: ['nfs-server', 'disk'], messages: [{ from: 'disk', to: 'nfs-server', network: '服务器本地（可选）', payload: 'train/image001.jpg 数据；也可能直接命中服务器缓存' }] },
      { title: '数据返回', description: '服务器将文件数据经 public 网络返回客户端内存，供应用读取。', activeNodes: ['nfs-server', 'public-switch', 'gpu-c1'], messages: [{ from: 'nfs-server', to: 'gpu-c1', network: 'public（经 public-switch）', payload: 'train/image001.jpg 数据 → 客户端内存' }] },
    ],
  },
  'ceph-read': {
    title: 'CephFS：元数据与数据读取',
    assumptions: [
      '示意文件的不同对象 A、B 分布在 OSD1、OSD2；不保证每个文件都这样分布。只展示 public 流量。',
      '客户端按需连接相关 MDS 和 OSD，而不是每次都连接所有节点；MON 不在每次读取的数据路径上，MDS 不转发文件内容。',
      'public 是客户端可达的存储网络，不等于互联网；读取不同对象的并发不等于同时读取同一对象的多个副本。',
    ],
    nodes: [
      { id: 'client', label: 'client', detail: 'CephFS 客户端' }, { id: 'mon', label: 'MON', detail: '认证与集群映射' },
      { id: 'mds', label: 'MDS', detail: '路径与元数据' }, { id: 'osd1', label: 'OSD1', detail: '对象 A' },
      { id: 'osd2', label: 'OSD2', detail: '对象 B' }, { id: 'osd3', label: 'OSD3', detail: '其他 OSD' },
    ],
    steps: [
      { title: '尚未挂载', description: '客户端尚无 CephFS 挂载，本例文件 /datasets/train.bin 尚未通过 CephFS 访问。', activeNodes: ['client'], messages: [] },
      { title: '认证并获取映射', description: '客户端联系 MON 完成认证并获取集群映射；MON 并非每次文件读都经过。', activeNodes: ['client', 'mon'], messages: [{ from: 'client', to: 'mon', network: 'public', payload: '认证、请求集群映射' }, { from: 'mon', to: 'client', network: 'public', payload: '认证结果与集群映射' }] },
      { title: '解析元数据', description: '客户端联系 MDS 解析路径、权限、文件 layout 与 caps（客户端能力授权）；MDS 不转发文件内容。', activeNodes: ['client', 'mds'], messages: [{ from: 'client', to: 'mds', network: 'public', payload: '解析 /datasets/train.bin 与访问权限' }, { from: 'mds', to: 'client', network: 'public', payload: '文件元数据、layout 与 caps' }] },
      { title: '定位对象', description: '客户端根据文件 layout 与集群映射定位本例的不同对象 A、B；这是客户端本地计算。', activeNodes: ['client'], messages: [{ from: 'client', to: 'client', network: '本地', payload: 'train.bin → 对象 A：OSD1；对象 B：OSD2' }] },
      { title: '并发读取不同对象', description: '客户端直接并发向 OSD1 和 OSD2 发起读取；A、B 是不同对象，不是同一对象的两个副本。', activeNodes: ['client', 'osd1', 'osd2'], messages: [{ from: 'client', to: 'osd1', network: 'public', payload: '读取对象 A' }, { from: 'client', to: 'osd2', network: 'public', payload: '读取对象 B' }] },
      { title: '数据直接返回客户端', description: 'OSD 将各自对象的数据直接返回客户端缓存或内存；不经过 MDS 转发。', activeNodes: ['client', 'osd1', 'osd2'], messages: [{ from: 'osd1', to: 'client', network: 'public', payload: '对象 A 数据 → 客户端缓存 / 内存' }, { from: 'osd2', to: 'client', network: 'public', payload: '对象 B 数据 → 客户端缓存 / 内存' }] },
    ],
  },
  'ceph-write': {
    title: 'CephFS：单个对象的三副本写入',
    assumptions: [
      '健康三副本池的简化示意：OSD1 是对象 A 的主 OSD，OSD2、OSD3 保存同一对象 A 的副本；不是通用 Ceph quorum 写入公式。',
      'cluster 表示配置独立后端网络时的复制流量；也可以只用 public 单网。public 是客户端可达的存储网络，不等于互联网。',
      '不同对象的条带化与同一对象的副本复制是两回事。普通 buffered write() 可能在存储层完成前返回；应用需要持久性时须使用 fsync 等适当同步机制。',
    ],
    nodes: [
      { id: 'client', label: 'client', detail: 'CephFS 客户端' }, { id: 'mon', label: 'MON', detail: '已有映射' },
      { id: 'mds', label: 'MDS', detail: '已有会话 / 权限' }, { id: 'osd1', label: 'OSD1', detail: '对象 A 的主 OSD' },
      { id: 'osd2', label: 'OSD2', detail: '对象 A 的副本' }, { id: 'osd3', label: 'OSD3', detail: '对象 A 的副本' },
    ],
    steps: [
      { title: '已有会话，选择对象', description: '假设客户端已有会话、权限与映射；从文件 /checkpoints/step-100.bin 选择其中对象 A。', activeNodes: ['client'], messages: [{ from: 'client', to: 'client', network: '本地', payload: '选择 step-100.bin 的对象 A' }] },
      { title: '写入主 OSD', description: '客户端根据映射，经 public 网络将对象 A 的写入发送给主 OSD1。', activeNodes: ['client', 'osd1'], messages: [{ from: 'client', to: 'osd1', network: 'public', payload: '写对象 A' }] },
      { title: '复制同一对象', description: '主 OSD1 可并行向两个副本 OSD 发送同一个对象 A 的写入；此处假设配置了独立 cluster 网络。', activeNodes: ['osd1', 'osd2', 'osd3'], messages: [{ from: 'osd1', to: 'osd2', network: 'cluster', payload: '复制对象 A' }, { from: 'osd1', to: 'osd3', network: 'cluster', payload: '复制对象 A' }] },
      { title: '副本持久化并确认', description: '两个副本各自持久化对象 A 的写入并向主 OSD 确认；两路可并行。', activeNodes: ['osd1', 'osd2', 'osd3'], messages: [{ from: 'osd2', to: 'osd1', network: 'cluster', payload: '对象 A 副本持久化确认' }, { from: 'osd3', to: 'osd1', network: 'cluster', payload: '对象 A 副本持久化确认' }] },
      { title: '主 OSD 汇集结果', description: '在此示意中，主 OSD1 自身满足持久化条件并汇集两个副本应答，准备报告存储层完成。', activeNodes: ['osd1'], messages: [{ from: 'osd1', to: 'osd1', network: '本地', payload: '对象 A：主副本持久化，汇集副本确认' }] },
      { title: '客户端收到完成', description: '主 OSD 向客户端报告本次对象 A 存储层完成；不要把这等同于普通 buffered write() 的返回时点。', activeNodes: ['osd1', 'client'], messages: [{ from: 'osd1', to: 'client', network: 'public', payload: '对象 A 存储层完成确认' }] },
    ],
  },
}
