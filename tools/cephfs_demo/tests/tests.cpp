#include "mini_cephfs.hpp"

#include <iostream>
#include <sstream>

using namespace mini_cephfs;

namespace {
std::size_t checks = 0;

// Deliberately not assert(): Release/NDEBUG must execute these checks too.
void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

struct Rig {
  Rados rados{4, 3, 4};
  Monitor monitor{rados};
  MDS mds;
  Client client{monitor, mds, rados};

  Rig() {
    check(client.connect(), "connect");
    check(client.mkdir("/data"), "mkdir");
    check(client.create("/data/file"), "create");
  }
};

void expect_text(Client& client, const std::string& expected) {
  const auto actual = client.read("/data/file", 0, std::numeric_limits<std::size_t>::max());
  check(actual && text(*actual) == expected, "file bytes differ: " + client.error());
  const auto info = client.stat("/data/file");
  check(info && info->size == expected.size(), "file size differs");
}

void check_replicas(const Rados& rados) {
  for (const auto& id : rados.known_objects()) {
    const auto nodes = rados.placement(id);
    const auto* first = rados.osd(nodes.front()).get(id);
    check(first && first->size() == rados.object_size(), "primary object size");
    for (const int node : nodes) {
      const auto* copy = rados.osd(node).get(id);
      check(copy && *copy == *first, "replica bytes differ");
    }
  }
}

void namespace_and_sessions() {
  Rig rig;
  auto& c = rig.client;
  Client disconnected(rig.monitor, rig.mds, rig.rados);
  check(!disconnected.create("/x") && !disconnected.mkdir("/x"), "disconnected namespace");
  check(!disconnected.stat("/") && !disconnected.ls("/"), "disconnected query");
  check(!disconnected.write("/data/file", 0, {}) && !disconnected.read("/data/file", 0, 1),
        "disconnected I/O");
  const auto root = c.stat("/");
  check(root && root->directory && root->inode == 1, "root inode");
  for (const std::string path : {"", "relative", "/data/", "/data//x", "/data/./x", "/data/../x"}) {
    check(!c.create(path) && !c.stat(path), "reject unsupported path: " + path);
  }
  check(!c.create(std::string("/data/a\0b", 9)), "reject embedded NUL");
  check(!c.mkdir("/") && !c.create("/data") && !c.create("/data/file"), "reject duplicates");
  check(!c.create("/missing/file") && !c.create("/data/file/child"), "parent must be directory");
  check(!c.stat("/data/file/child"), "lookup cannot cross a file");
  check(!c.write("/data", 0, bytes("x")) && !c.read("/data", 0, 1), "reject directory I/O");
  check(!c.ls("/data/file") && !c.ls("/missing"), "ls failure distinct from empty list");
  check(c.mkdir("/data/nested"), "nested directory");
  const auto empty = c.ls("/data/nested");
  check(empty && empty->empty() && c.error().empty(), "successful empty ls clears error");
  const auto names = c.ls("/data");
  check(names && *names == std::vector<std::string>({"file", "nested"}), "sorted directory entries");
}

void boundaries_and_binary_data() {
  Rig rig;
  auto& c = rig.client;
  expect_text(c, "");
  check(c.write("/data/file", 0, {}), "empty write");
  check(rig.rados.known_objects().empty(), "empty file has no objects");
  check(c.write("/data/file", 0, bytes("abcdefghij")), "write three objects");
  check(rig.rados.known_objects().size() == 3, "object count");
  const auto partial = c.read("/data/file", 2, 100);
  check(partial && text(*partial) == "cdefghij", "truncate read at EOF");
  check(c.write("/data/file", 3, bytes("XYZ")), "cross-object overwrite");
  expect_text(c, "abcXYZghij");
  check(c.write("/data/file", 10, bytes("kl")), "append to partial final object");
  check(c.write("/data/file", 12, bytes("m")), "append at object boundary");
  expect_text(c, "abcXYZghijklm");
  for (const auto offset : {std::size_t{13}, std::numeric_limits<std::size_t>::max()}) {
    const auto eof = c.read("/data/file", offset, 1);
    check(eof && eof->empty(), "read at/beyond EOF");
  }
  const auto zero = c.read("/data/file", 1, 0);
  check(zero && zero->empty(), "zero read");
  check(!c.write("/data/file", 14, bytes("gap")) && !c.write("/data/file", 14, {}), "no sparse writes");
  check(c.write("/data/file", 2, {}), "zero write does not truncate");
  expect_text(c, "abcXYZghijklm");
  check(c.create("/data/binary"), "second file");
  const Bytes binary{0, 255, 127, 0, 200};
  check(c.write("/data/binary", 0, binary), "binary write");
  const auto copy = c.read("/data/binary", 0, 5);
  check(copy && *copy == binary, "binary read");
  expect_text(c, "abcXYZghijklm");
  check_replicas(rig.rados);
}

void placement_and_configuration() {
  for (const auto& config : {std::vector<std::size_t>{0, 1, 4}, {4, 0, 4}, {4, 5, 4}, {4, 3, 0}}) {
    bool rejected = false;
    try { Rados invalid(config[0], config[1], config[2]); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid configuration");
  }
  Rig rig;
  for (std::size_t index = 0; index < 10; ++index) {
    const ObjectId id{3, index};
    const auto targets = rig.rados.placement(id);
    check(targets == std::vector<int>({static_cast<int>((3 + index) % 4),
                                      static_cast<int>((4 + index) % 4),
                                      static_cast<int>((5 + index) % 4)}), "deterministic placement");
    check(std::set<int>(targets.begin(), targets.end()).size() == 3, "distinct replica targets");
  }
  Bytes output;
  check(!rig.rados.read({999, 0}, output), "missing object is an error");
  check(!rig.rados.write({999, 0}, bytes("x")), "object size is enforced");
  const auto inode = rig.client.stat("/data/file")->inode;
  // Synthetic metadata allows the arithmetic guard to be reached without huge allocations.
  check(rig.mds.update_size(inode, std::numeric_limits<std::size_t>::max()), "set synthetic size");
  check(!rig.client.write("/data/file", std::numeric_limits<std::size_t>::max(), bytes("x")),
        "offset + input length overflow");
  check(rig.rados.known_objects().empty(), "overflow rejected before any data mutation");
}

void outage_and_repair() {
  Rig rig;
  auto& c = rig.client;
  check(c.write("/data/file", 0, bytes("abcdefgh")), "initial data");
  const auto first = *c.known_objects().begin();
  const int primary = c.placement(first).front();
  c.stop_osd(primary);
  expect_text(c, "abcdefgh");
  check(!c.write("/data/file", 0, bytes("X")), "reject write with offline target");
  expect_text(c, "abcdefgh");
  check(!c.repair(), "repair requires online targets");
  c.start_osd(primary);  // retained objects; no writes were accepted while absent.
  check(c.write("/data/file", 0, bytes("A")), "restart retained node");
  check(c.replace_osd(primary), "empty replacement");
  check(!rig.rados.osd(primary).has(first), "replacement really lost memory");
  expect_text(c, "Abcdefgh");
  c.stop_osd(primary);
  c.start_osd(primary);
  check(!c.write("/data/file", 0, bytes("Z")), "restart cannot bypass recovery");
  check(c.repair(), "copy surviving replicas to replacement");
  check_replicas(rig.rados);
  check(c.write("/data/file", 8, bytes("!")), "writes resume after repair");
  expect_text(c, "Abcdefgh!");
  for (const int node : c.placement(first)) c.stop_osd(node);
  check(!c.read("/data/file", 0, 1), "all replicas unavailable is not empty data");
}

void whole_write_preflight() {
  Rig rig;
  auto& c = rig.client;
  check(c.write("/data/file", 0, bytes("abcdefgh")), "initial data");
  const auto original_objects = c.known_objects();
  auto object = original_objects.begin();
  const auto first_nodes = c.placement(*object++);
  const auto second_nodes = c.placement(*object);
  int failing_node = -1;
  for (const int node : second_nodes) {
    if (std::find(first_nodes.begin(), first_nodes.end(), node) == first_nodes.end()) failing_node = node;
  }
  check(failing_node >= 0, "second object has a target absent from first");
  c.stop_osd(failing_node);
  check(rig.rados.can_write(*original_objects.begin()), "first object alone is writable");
  check(!c.write("/data/file", 3, bytes("0123456789")), "later object preflight fails");
  expect_text(c, "abcdefgh");
  check(c.known_objects() == original_objects, "no new object created on rejected write");
  check(c.replace_osd(failing_node), "replace later target");
  check(!c.write("/data/file", 3, bytes("0123456789")), "recovery state rejects whole write");
  expect_text(c, "abcdefgh");
  check(c.repair(), "repair later target");
  check(c.write("/data/file", 3, bytes("0123456789")), "whole write succeeds after repair");
  expect_text(c, "abc0123456789");
  check_replicas(rig.rados);
}

void permanent_loss() {
  Rig rig;
  auto& c = rig.client;
  check(c.write("/data/file", 0, bytes("lost data")), "initial data");
  const auto id = *c.known_objects().begin();
  for (const int node : c.placement(id)) check(c.replace_osd(node), "replace every replica");
  check(!c.read("/data/file", 0, 1), "online but empty targets do not fabricate data");
  check(!c.repair(), "repair cannot recreate all-lost content");
  for (const int node : c.placement(id)) {
    check(!rig.rados.osd(node).accepts_writes() && !rig.rados.osd(node).has(id),
          "failed repair does not admit empty nodes");
  }
  check(!c.write("/data/file", 0, bytes("new")), "lost existing data requires explicit error");
}

void operation_trace() {
  std::ostringstream log;
  Rados rados(4, 3, 4, &log);
  Monitor mon(rados);
  MDS mds(&log);
  Client c(mon, mds, rados);
  check(c.connect() && c.create("/file") && c.write("/file", 0, bytes("hello")), "traced write");
  check(c.read("/file", 0, 5).has_value(), "traced read");
  check(log.str().find("Client -> MDS: CREATE /file") != std::string::npos, "metadata trace");
  check(log.str().find("REPLICATE ino-2.obj-0") != std::string::npos, "primary replication trace");
  check(log.str().find("GET ino-2.obj-1") != std::string::npos, "data trace");
}
}  // namespace

int main() {
  const std::vector<std::pair<std::string, void (*)()>> tests{
      {"namespace_and_sessions", namespace_and_sessions}, {"boundaries_and_binary_data", boundaries_and_binary_data},
      {"placement_and_configuration", placement_and_configuration}, {"outage_and_repair", outage_and_repair},
      {"whole_write_preflight", whole_write_preflight}, {"permanent_loss", permanent_loss},
      {"operation_trace", operation_trace}};
  int failures = 0;
  for (const auto& test : tests) {
    try {
      test.second();
      std::cout << "PASS " << test.first << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
    }
  }
  std::cout << checks << " checks, " << failures << " failing groups\n";
  return failures == 0 ? 0 : 1;
}
