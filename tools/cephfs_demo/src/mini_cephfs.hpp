#pragma once

// Mini CephFS is a single-process teaching model. It is deliberately not
// wire-compatible with Ceph and stores every byte in memory.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mini_cephfs {

using Bytes = std::vector<std::uint8_t>;

inline Bytes bytes(const std::string& text) {
  return Bytes(text.begin(), text.end());
}

inline std::string text(const Bytes& data) {
  return std::string(data.begin(), data.end());
}

struct ObjectId {
  std::uint64_t inode = 0;
  std::size_t index = 0;

  bool operator<(const ObjectId& other) const {
    return std::pair{inode, index} < std::pair{other.inode, other.index};
  }
  bool operator==(const ObjectId& other) const {
    return inode == other.inode && index == other.index;
  }
};

inline std::string object_name(const ObjectId& id) {
  return "ino-" + std::to_string(id.inode) + ".obj-" + std::to_string(id.index);
}

class OSD {
 public:
  explicit OSD(int id) : id_(id) {}

  int id() const { return id_; }
  bool online() const { return online_; }
  bool accepts_writes() const { return online_ && accepts_writes_; }

  void stop() { online_ = false; }
  void start() { online_ = true; }  // restarting does not bypass pending recovery.
  void replace_empty() {
    objects_.clear();
    online_ = true;
    accepts_writes_ = false;  // repair must populate it before normal writes.
  }

  bool has(const ObjectId& id) const {
    return objects_.find(id) != objects_.end();
  }

  const Bytes* get(const ObjectId& id) const {
    const auto it = objects_.find(id);
    return it == objects_.end() ? nullptr : &it->second;
  }

  bool put(const ObjectId& id, const Bytes& data, bool recovery = false) {
    if (!online_ || (!accepts_writes_ && !recovery)) return false;
    objects_[id] = data;
    return true;
  }

  // Called on the primary after Rados has preflighted every target.
  void replicate(const ObjectId& id, const Bytes& data, const std::vector<OSD*>& peers,
                 std::ostream* trace) {
    if (!put(id, data)) throw std::logic_error("primary changed after preflight");
    if (trace) *trace << "  Client -> OSD" << id_ << ": PUT " << object_name(id) << '\n';
    for (OSD* peer : peers) {
      if (!peer->put(id, data)) throw std::logic_error("replica changed after preflight");
      if (trace) *trace << "  OSD" << id_ << " -> OSD" << peer->id()
                        << ": REPLICATE " << object_name(id) << '\n';
    }
  }

  void allow_writes() { accepts_writes_ = true; }

 private:
  int id_;
  bool online_ = true;
  bool accepts_writes_ = true;
  std::map<ObjectId, Bytes> objects_;
};

class Rados {
 public:
  Rados(std::size_t osd_count, std::size_t replicas, std::size_t object_size,
        std::ostream* trace = nullptr)
      : replicas_(replicas), object_size_(object_size), trace_(trace) {
    if (osd_count == 0 || osd_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        replicas == 0 || replicas > osd_count || object_size == 0 ||
        object_size > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
      throw std::invalid_argument("invalid RADOS teaching-model configuration");
    }
    for (std::size_t i = 0; i < osd_count; ++i) osds_.emplace_back(static_cast<int>(i));
  }

  std::size_t object_size() const { return object_size_; }
  std::size_t osd_count() const { return osds_.size(); }
  OSD& osd(int id) { return osds_.at(static_cast<std::size_t>(id)); }
  const OSD& osd(int id) const { return osds_.at(static_cast<std::size_t>(id)); }

  std::vector<int> placement(const ObjectId& id) const {
    std::vector<int> result;
    // Reduce each term first to avoid overflow; this is NOT CRUSH.
    const std::size_t primary = static_cast<std::size_t>(
        (id.inode % osds_.size() + id.index % osds_.size()) % osds_.size());
    for (std::size_t i = 0; i < replicas_; ++i) {
      result.push_back(static_cast<int>((primary + i) % osds_.size()));
    }
    return result;
  }

  bool can_write(const ObjectId& id, std::string* error = nullptr) const {
    const Bytes* baseline = nullptr;
    for (const int osd_id : placement(id)) {
      const OSD& node = osd(osd_id);
      if (!node.accepts_writes()) {
        if (error) *error = "OSD" + std::to_string(osd_id) + " is unavailable for writes";
        return false;
      }
      if (known_objects_.count(id) != 0) {
        const Bytes* data = node.get(id);
        if (!data || data->size() != object_size_ || (baseline && *data != *baseline)) {
          if (error) *error = "missing or inconsistent replica: " + object_name(id);
          return false;
        }
        baseline = data;
      }
    }
    return true;
  }

  bool write(const ObjectId& id, const Bytes& data, std::string* error = nullptr) {
    if (data.size() != object_size_) {
      if (error) *error = "object must contain exactly object_size bytes";
      return false;
    }
    if (!can_write(id, error)) return false;
    const auto targets = placement(id);
    std::vector<OSD*> peers;
    for (std::size_t i = 1; i < targets.size(); ++i) peers.push_back(&osd(targets[i]));
    osd(targets.front()).replicate(id, data, peers, trace_);
    known_objects_.insert(id);
    return true;
  }

  bool read(const ObjectId& id, Bytes& out, std::string* error = nullptr) const {
    for (const int osd_id : placement(id)) {
      const OSD& node = osd(osd_id);
      if (!node.online()) continue;
      const Bytes* data = node.get(id);
      if (data != nullptr && data->size() == object_size_) {
        if (trace_) *trace_ << "  Client -> OSD" << osd_id << ": GET " << object_name(id) << '\n';
        out = *data;
        return true;
      }
    }
    if (error) *error = "no readable replica for " + object_name(id);
    return false;
  }

  bool repair_all(std::string* error = nullptr) {
    // The in-memory catalog survives an OSD replacement, NOT a process restart.
    // Check every source/target before admitting replacement nodes for writes.
    std::map<ObjectId, Bytes> sources;
    for (const OSD& node : osds_) {
      if (!node.online()) {
        if (error) *error = "repair requires all configured OSDs online";
        return false;
      }
    }
    for (const ObjectId& id : known_objects_) {
      if (!read(id, sources[id], error)) return false;
    }
    for (const auto& item : sources) {
      for (const int osd_id : placement(item.first)) {
        OSD& node = osd(osd_id);
        if (!node.has(item.first)) {
          node.put(item.first, item.second, true);
          if (trace_) *trace_ << "  REPAIR -> OSD" << osd_id << ": "
                              << object_name(item.first) << '\n';
        }
      }
    }
    for (OSD& node : osds_) node.allow_writes();
    return true;
  }

  const std::set<ObjectId>& known_objects() const { return known_objects_; }

 private:
  std::vector<OSD> osds_;
  std::size_t replicas_;
  std::size_t object_size_;
  std::ostream* trace_;
  std::set<ObjectId> known_objects_;
};

class Monitor {
 public:
  explicit Monitor(const Rados& rados) : rados_(rados) {}
  bool connect() const { return rados_.osd_count() > 0; }
  std::string cluster_view() const {
    return "static teaching map: " + std::to_string(rados_.osd_count()) + " OSDs";
  }

 private:
  const Rados& rados_;
};

struct FileStat {
  std::uint64_t inode = 0;
  std::size_t size = 0;
  bool directory = false;
};

class MDS {
 private:
  struct Entry {
    std::uint64_t inode = 0;
    bool directory = false;
    std::size_t size = 0;
    std::map<std::string, std::uint64_t> children;
  };

 public:
  explicit MDS(std::ostream* trace = nullptr) : trace_(trace) {
    entries_.emplace(1, Entry{1, true, 0, {}});
  }

  std::optional<FileStat> stat(const std::string& path, std::string* error = nullptr) const {
    const auto inode = resolve(path, error);
    if (!inode) return std::nullopt;
    const Entry& entry = entries_.at(*inode);
    return FileStat{entry.inode, entry.size, entry.directory};
  }

  bool mkdir(const std::string& path, std::string* error = nullptr) {
    return create_entry(path, true, error);
  }

  bool create(const std::string& path, std::string* error = nullptr) {
    return create_entry(path, false, error);
  }

  bool update_size(std::uint64_t inode, std::size_t size, std::string* error = nullptr) {
    const auto it = entries_.find(inode);
    if (it == entries_.end() || it->second.directory) {
      if (error) *error = "cannot update size of missing directory/inode";
      return false;
    }
    it->second.size = size;
    if (trace_) *trace_ << "  Client -> MDS: size inode=" << inode << " -> " << size << '\n';
    return true;
  }

  std::optional<std::vector<std::string>> list(const std::string& path,
                                               std::string* error = nullptr) const {
    const auto inode = resolve(path, error);
    if (!inode) return std::nullopt;
    const Entry& entry = entries_.at(*inode);
    if (!entry.directory) {
      if (error) *error = "not a directory: " + path;
      return std::nullopt;
    }
    std::vector<std::string> names;
    for (const auto& child : entry.children) names.push_back(child.first);
    return names;
  }

 private:
  static std::optional<std::vector<std::string>> split(const std::string& path, std::string* error) {
    if (path == "/") return std::vector<std::string>{};
    if (path.empty() || path.front() != '/' || path.back() == '/' ||
        path.find('\0') != std::string::npos) {
      if (error) *error = "path must be an absolute normalized path: " + path;
      return std::nullopt;
    }
    std::vector<std::string> parts;
    std::size_t start = 1;
    while (start < path.size()) {
      const std::size_t end = path.find('/', start);
      const std::string part = path.substr(start, end == std::string::npos ? end : end - start);
      if (part.empty() || part == "." || part == "..") {
        if (error) *error = "path contains an unsupported component: " + path;
        return std::nullopt;
      }
      parts.push_back(part);
      if (end == std::string::npos) break;
      start = end + 1;
    }
    return parts;
  }

  std::optional<std::uint64_t> resolve(const std::string& path, std::string* error) const {
    if (trace_) *trace_ << "  Client -> MDS: LOOKUP " << path << '\n';
    const auto parts = split(path, error);
    if (!parts) return std::nullopt;
    std::uint64_t current = 1;
    for (const std::string& part : *parts) {
      if (!entries_.at(current).directory) {
        if (error) *error = "path crosses a non-directory: " + path;
        return std::nullopt;
      }
      const auto child = entries_.at(current).children.find(part);
      if (child == entries_.at(current).children.end()) {
        if (error) *error = "path does not exist: " + path;
        return std::nullopt;
      }
      current = child->second;
    }
    return current;
  }

  bool create_entry(const std::string& path, bool directory, std::string* error) {
    if (trace_) *trace_ << "  Client -> MDS: " << (directory ? "MKDIR " : "CREATE ") << path << '\n';
    const auto parts = split(path, error);
    if (!parts || parts->empty()) {
      if (error && parts && parts->empty()) *error = "cannot create the root directory";
      return false;
    }
    const std::string leaf = parts->back();
    std::string parent_path = "/";
    for (std::size_t i = 0; i + 1 < parts->size(); ++i) parent_path += (*parts)[i] + "/";
    if (parent_path.size() > 1) parent_path.pop_back();
    const auto parent = resolve(parent_path, error);
    if (!parent) return false;
    Entry& parent_entry = entries_.at(*parent);
    if (!parent_entry.directory) {
      if (error) *error = "parent is not a directory: " + parent_path;
      return false;
    }
    if (parent_entry.children.count(leaf) != 0) {
      if (error) *error = "path already exists: " + path;
      return false;
    }
    const std::uint64_t inode = next_inode_++;
    entries_.emplace(inode, Entry{inode, directory, 0, {}});
    parent_entry.children.emplace(leaf, inode);
    return true;
  }

  std::map<std::uint64_t, Entry> entries_;
  std::uint64_t next_inode_ = 2;
  std::ostream* trace_ = nullptr;
};

class Client {
 public:
  Client(Monitor& monitor, MDS& mds, Rados& rados)
      : monitor_(monitor), mds_(mds), rados_(rados) {}

  bool connect() {
    error_.clear();
    connected_ = monitor_.connect();
    if (!connected_) error_ = "MON returned no cluster view";
    return connected_;
  }
  bool connected() const { return connected_; }
  const std::string& error() const { return error_; }

  bool mkdir(const std::string& path) { return connected_call([&] { return mds_.mkdir(path, &error_); }); }
  bool create(const std::string& path) { return connected_call([&] { return mds_.create(path, &error_); }); }

  bool write(const std::string& path, std::size_t offset, const Bytes& input) {
    error_.clear();
    if (!connected_) return fail("client is not connected");
    const auto stat = mds_.stat(path, &error_);
    if (!stat) return false;
    if (stat->directory) return fail("path is a directory: " + path);
    if (offset > stat->size || input.size() > std::numeric_limits<std::size_t>::max() - offset) {
      return fail("write would create a gap or overflow");
    }
    if (input.empty()) return true;

    const std::size_t end = offset + input.size();
    const std::size_t first = offset / rados_.object_size();
    const std::size_t last = (end - 1) / rados_.object_size();
    struct Pending {
      ObjectId id;
      Bytes data;
    };
    std::vector<Pending> pending;

    for (std::size_t index = first; index <= last; ++index) {
      const ObjectId id{stat->inode, index};
      Bytes object(rados_.object_size(), 0);
      const std::size_t object_start = index * rados_.object_size();
      if (object_start < stat->size && !rados_.read(id, object, &error_)) return false;
      const std::size_t begin = std::max(offset, object_start);
      const std::size_t finish = object_start + std::min(end - object_start, rados_.object_size());
      for (std::size_t position = begin; position < finish; ++position) {
        object[position - object_start] = input[position - offset];
      }
      if (!rados_.can_write(id, &error_)) return false;
      pending.push_back(Pending{id, std::move(object)});
    }

    for (const Pending& item : pending) {
      if (!rados_.write(item.id, item.data, &error_)) return false;
    }
    const std::size_t new_size = std::max(stat->size, end);
    return mds_.update_size(stat->inode, new_size, &error_);
  }

  std::optional<Bytes> read(const std::string& path, std::size_t offset, std::size_t length) {
    error_.clear();
    if (!connected_) {
      fail("client is not connected");
      return std::nullopt;
    }
    const auto stat = mds_.stat(path, &error_);
    if (!stat) return std::nullopt;
    if (stat->directory) {
      fail("path is a directory: " + path);
      return std::nullopt;
    }
    if (offset >= stat->size || length == 0) return Bytes{};
    const std::size_t available = stat->size - offset;
    const std::size_t wanted = std::min(length, available);
    const std::size_t end = offset + wanted;
    Bytes result;
    result.reserve(wanted);
    for (std::size_t index = offset / rados_.object_size();
         index <= (end - 1) / rados_.object_size(); ++index) {
      Bytes object;
      if (!rados_.read(ObjectId{stat->inode, index}, object, &error_)) return std::nullopt;
      const std::size_t object_start = index * rados_.object_size();
      const std::size_t begin = std::max(offset, object_start);
      const std::size_t finish = object_start + std::min(end - object_start, rados_.object_size());
      result.insert(result.end(), object.begin() + static_cast<std::ptrdiff_t>(begin - object_start),
                    object.begin() + static_cast<std::ptrdiff_t>(finish - object_start));
    }
    return result;
  }

  std::optional<FileStat> stat(const std::string& path) {
    error_.clear();
    if (!connected_) {
      fail("client is not connected");
      return std::nullopt;
    }
    return mds_.stat(path, &error_);
  }

  std::optional<std::vector<std::string>> ls(const std::string& path) {
    error_.clear();
    if (!connected_) {
      fail("client is not connected");
      return std::nullopt;
    }
    return mds_.list(path, &error_);
  }

  bool replace_osd(int id) {
    error_.clear();
    if (!connected_) return fail("client is not connected");
    if (id < 0 || static_cast<std::size_t>(id) >= rados_.osd_count()) {
      return fail("OSD id is out of range");
    }
    rados_.osd(id).replace_empty();
    return true;
  }
  void stop_osd(int id) { rados_.osd(id).stop(); }
  void start_osd(int id) { rados_.osd(id).start(); }

  bool repair() {
    error_.clear();
    if (!connected_) return fail("client is not connected");
    return rados_.repair_all(&error_);
  }

  std::vector<int> placement(const ObjectId& id) const { return rados_.placement(id); }
  std::size_t object_size() const { return rados_.object_size(); }
  const std::set<ObjectId>& known_objects() const { return rados_.known_objects(); }

 private:
  template <typename Function>
  bool connected_call(Function&& function) {
    error_.clear();
    if (!connected_) return fail("client is not connected");
    return function();
  }
  bool fail(const std::string& message) {
    error_ = message;
    return false;
  }

  Monitor& monitor_;
  MDS& mds_;
  Rados& rados_;
  bool connected_ = false;
  std::string error_;
};

}  // namespace mini_cephfs
