# 🌐 Distributed Systems

## 笔记索引

- [分布式系统核心理论基础](./distributed-systems-foundations.md) — 初学者教程：从共享笔记的保存、旧读、断网和换主出发，再理解 CAP/FLP、共识、Quorum、时钟、分片和事务

## Distributed Storage

- [分布式存储基础总览](./distributed-storage/index.md) — 给初学者的总入口：先建立分布式存储的整体框架
- [面向 AI 训练与多云场景的存储工程地图](./distributed-storage/ai-training-and-multicloud-storage-map.md) — 多级缓存、RDMA / NVMe-oF、Kubernetes 与多模态数据场景
- [对象存储](./distributed-storage/object-storage.md) — Bucket/Object/Key、S3、Multipart、生命周期、一致性
- [并行文件系统](./distributed-storage/parallel-file-systems.md) — POSIX 语义、元数据、条带化、锁、Lustre/CephFS/GPFS
- [NFS：部署、挂载与计算节点连接](./distributed-storage/nfs-deployment-and-mount.md) — 导出、路径映射、网络与挂载读取交互演示
- [CephFS：并行访问、部署、挂载与网络](./distributed-storage/cephfs-deployment-and-mount.md) — MON/MDS/OSD、双网络、cephadm 与读写路径交互演示
- [副本、纠删码与放置](./distributed-storage/replication-erasure-coding-placement.md) — 复制、EC、故障域、CRUSH、重平衡
- [一致性、共识与故障恢复](./distributed-storage/consistency-consensus-failure.md) — 一致性、Quorum、Raft、租约、fencing、脑裂
- [元数据与性能瓶颈](./distributed-storage/metadata-and-performance.md) — 小文件、热点、尾延迟、缓存、重平衡
- [块存储、文件存储与对象存储对比](./distributed-storage/block-file-object-comparison.md) — 三种存储语义、接口、场景的直观对照
- [Ceph、Lustre、MinIO 入门概览](./distributed-storage/ceph-lustre-minio-overview.md) — 常见系统定位、结构和学习入口
- [目录总览](./distributed-storage/) — 这一组笔记的入口目录

## Databases

- LSM-Tree
- B-Tree
- Transactions
- Isolation Levels
