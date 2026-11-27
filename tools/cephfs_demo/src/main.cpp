#include "mini_cephfs.hpp"

#include <iostream>
#include <string>

using mini_cephfs::Client;
using mini_cephfs::MDS;
using mini_cephfs::Monitor;
using mini_cephfs::Rados;
using mini_cephfs::bytes;
using mini_cephfs::text;

namespace {

void show_placement(const Client& client) {
  std::cout << "对象布局（固定 3 副本；不是 CRUSH/PG）：\n";
  for (const auto& id : client.known_objects()) {
    std::cout << "  " << mini_cephfs::object_name(id) << " -> ";
    const auto nodes = client.placement(id);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      if (i != 0) std::cout << ", ";
      std::cout << "OSD" << nodes[i];
    }
    std::cout << '\n';
  }
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    Rados rados(/*osds=*/4, /*replicas=*/3, /*object_size=*/8, &std::cout);
    Monitor monitor(rados);
    MDS mds(&std::cout);
    Client client(monitor, mds, rados);

    require(client.connect(), client.error());
    std::cout << "Client -> MON: " << monitor.cluster_view() << '\n';
    require(client.mkdir("/datasets"), client.error());
    require(client.mkdir("/datasets/train"), client.error());
    require(client.create("/datasets/train/sample.bin"), client.error());

    const auto payload = bytes("CephFS teaching model: objects + replicas");
    require(client.write("/datasets/train/sample.bin", 0, payload), client.error());
    const auto stat = client.stat("/datasets/train/sample.bin");
    require(stat.has_value(), client.error());
    std::cout << "MDS: inode=" << stat->inode << ", size=" << stat->size << " bytes\n";
    show_placement(client);

    const auto initial = client.read("/datasets/train/sample.bin", 0, 1000);
    require(initial.has_value(), client.error());
    require(text(*initial) == text(payload), "initial read mismatch");
    std::cout << "Client -> OSD: 读取并重组成功：" << text(*initial) << '\n';

    require(client.write("/datasets/train/sample.bin", 7, bytes("[OBJ]")), client.error());
    const auto patched = client.read("/datasets/train/sample.bin", 0, 1000);
    require(patched.has_value(), client.error());
    std::string expected = text(payload);
    expected.replace(7, 5, "[OBJ]");
    require(text(*patched) == expected, "cross-object overwrite mismatch");
    std::cout << "跨对象覆盖后：" << text(*patched) << '\n';

    const auto first_object = *client.known_objects().begin();
    const int failed_primary = client.placement(first_object).front();
    std::cout << "故障注入：停止 OSD" << failed_primary << "，读取应从副本回退\n";
    client.stop_osd(failed_primary);
    const auto degraded = client.read("/datasets/train/sample.bin", 0, 1000);
    require(degraded.has_value(), client.error());
    require(*degraded == *patched, "degraded read mismatch");
    require(!client.write("/datasets/train/sample.bin", 0, bytes("!")),
            "write should fail while a replica target is offline");
    std::cout << "降级读取成功；写入被拒绝：" << client.error() << '\n';
    const auto unchanged = client.read("/datasets/train/sample.bin", 0, 1000);
    const auto unchanged_stat = client.stat("/datasets/train/sample.bin");
    require(unchanged && *unchanged == *patched && unchanged_stat &&
                unchanged_stat->size == patched->size(), "failed write changed file");

    require(client.replace_osd(failed_primary), client.error());
    require(!client.write("/datasets/train/sample.bin", 0, bytes("!")),
            "replacement must remain read-only until repair");
    require(client.repair(), client.error());
    std::cout << "OSD" << failed_primary << " 已替换为空节点，repair() 完成\n";
    require(client.write("/datasets/train/sample.bin", patched->size(), bytes("!")), client.error());
    const auto recovered = client.read("/datasets/train/sample.bin", 0, 1000);
    require(recovered.has_value(), client.error());
    require(text(*recovered) == expected + "!", "repaired append mismatch");
    for (const auto& id : client.known_objects()) {
      const auto nodes = client.placement(id);
      for (const int node : nodes) {
        const auto* copy = rados.osd(node).get(id);
        require(copy && *copy == *rados.osd(nodes.front()).get(id), "replica mismatch");
      }
    }
    std::cout << "修复后读取：" << text(*recovered) << '\n';
    for (const int node : client.placement(first_object)) client.stop_osd(node);
    require(!client.read("/datasets/train/sample.bin", 0, 1), "lost replicas should fail");
    std::cout << "全部副本不可读：" << client.error() << '\n';
    std::cout << "演示完成：这是 CephFS 核心路径模型，不是生产 CephFS 实现。\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "demo failed: " << error.what() << '\n';
    return 1;
  }
}
