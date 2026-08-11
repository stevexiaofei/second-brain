---
title: Git 在 NFS 上的性能问题
type: experience
status: growing
tags: [Git, NFS, Linux]
---

# Git 在 NFS 上的性能问题

## Problem

在 NFS 磁盘上的 Git 仓库执行 `git status` 可能明显变慢。

## Investigation

重点观察：

- NFS RTT
- metadata latency
- inode / dentry cache
- mount options
- Git 是否扫描大量未跟踪文件
- 仓库中文件数量
- `.git/index`

## Root Cause

NFS 对大量小文件和 metadata 操作通常比本地文件系统敏感。Git 的工作区扫描会放大这种延迟。

## Lessons Learned

遇到 NFS + Git 性能问题，不要只看带宽；重点检查 metadata、RTT、cache、file count 和 Git working-tree scan。
