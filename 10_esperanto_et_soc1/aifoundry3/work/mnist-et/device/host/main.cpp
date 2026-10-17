// mnist on ET-SoC-1: for each call of a run directory (harness/release.py make), copy the draw to the card,
// train and label in one launch, copy the labels back. Writes <out>/call_NN.labels and <out>/times.json for
// release.py score, and <out>/stats.json with the kernel's per-phase cycle counts.
//   mnist_host --run DIR --out DIR [--sysemu [--sim-args "..."]] [--shires MASK] [--calls warmup,0,1|all]
//              [--steps N] [--n-train N] [--n-test N] [--wout] [--budget S] [--poll-limit N]
// The card is opened once (the untimed set-up, like compiling on the A100); the warm-up call is untimed.
// Host structure from ~/et-soc1-prototyping/workloads/sgemm/host/main.cpp.
#include <g3log/loglevels.hpp>
#include <runtime/IRuntime.h>
#include <runtime/Types.h>
#include <device-layer/IDeviceLayer.h>
#include <sw-sysemu/SysEmuOptions.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "Constants.h"
#include "mnist_args.h"

// The runtime's logging race (docs/findings/14-card-behaviour.md, "Traps"): register its levels before any thread.
static void registerRuntimeLogLevels() {
  const LEVELS levels[] = {LEVELS(g3::kDebugValue - 100, "VERBOSE_HIGH"), LEVELS(g3::kDebugValue - 99, "VERBOSE_MID"),
                           LEVELS(g3::kDebugValue - 98, "VERBOSE_LOW")};
  for (const auto& l : levels) g3::only_change_at_initialization::addLogLevel(l, false);
}

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static double msSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::vector<std::byte> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return {};
  std::vector<std::byte> data(fs::file_size(path));
  file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return data;
}

static std::unique_ptr<dev::IDeviceLayer> makeDeviceLayer(bool sysemu, const std::string& simArgs) {
  if (!sysemu) return dev::IDeviceLayer::createPcieDeviceLayer(true, true);
  emu::SysEmuOptions o;
  o.bootromTrampolineToBL2ElfPath = BOOTROM_TRAMPOLINE_TO_BL2_ELF;
  o.spBL2ElfPath = BL2_ELF;
  o.masterMinionElfPath = MASTER_MINION_ELF;
  o.machineMinionElfPath = MACHINE_MINION_ELF;
  o.workerMinionElfPath = WORKER_MINION_ELF;
  o.executablePath = fs::path(SYSEMU_INSTALL_DIR) / "sys_emu";
  o.runDir = fs::current_path();
  o.maxCycles = std::numeric_limits<uint64_t>::max();
  o.minionShiresMask = 0x1FFFFFFFFu;
  o.puUart0Path = o.runDir + "/pu_uart0_tx.log";
  o.puUart1Path = o.runDir + "/pu_uart1_tx.log";
  o.spUart0Path = o.runDir + "/spio_uart0_tx.log";
  o.spUart1Path = o.runDir + "/spio_uart1_tx.log";
  o.startGdb = false;
  std::istringstream extra(simArgs);
  for (std::string arg; extra >> arg;) o.additionalOptions.push_back(arg);
  return dev::IDeviceLayer::createSysEmuDeviceLayer(o, 1);
}

static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

constexpr uint32_t kTrain = 10000, kTest = 10000, kDin = 60;

struct Call {
  std::string name;  // "warmup" or "00" .. "14"
  std::vector<float> x, tx;
  std::vector<int32_t> y;
};

static bool loadCall(const fs::path& path, Call& c) {
  const auto raw = readFile(path.string());
  const size_t want = (size_t(kTrain) * kDin + kTrain + size_t(kTest) * kDin) * 4;
  if (raw.size() != want) return false;
  c.x.resize(size_t(kTrain) * kDin);
  c.y.resize(kTrain);
  c.tx.resize(size_t(kTest) * kDin);
  const std::byte* p = raw.data();
  std::memcpy(c.x.data(), p, c.x.size() * 4); p += c.x.size() * 4;
  std::memcpy(c.y.data(), p, c.y.size() * 4); p += c.y.size() * 4;
  std::memcpy(c.tx.data(), p, c.tx.size() * 4);
  return true;
}

int main(int argc, char** argv) {
  registerRuntimeLogLevels();  // first, before any library starts a thread
  bool sysemu = false, wout = false;
  uint64_t shireMask = 0xffffffff;
  uint32_t steps = 100, nTrain = kTrain, nTest = kTest;
  uint64_t pollLimit = 0;
  double budgetS = -1;
  std::string runDir, outDir, callsArg = "all", simArgs, kernelPath = KERNEL_ELF;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { std::cerr << a << " needs a value\n"; std::exit(2); }
      return argv[++i];
    };
    if (a == "--sysemu") sysemu = true;
    else if (a == "--sim-args") simArgs = next();
    else if (a == "--shires") shireMask = std::strtoull(next(), nullptr, 0) & 0xffffffffull;
    else if (a == "--run") runDir = next();
    else if (a == "--out") outDir = next();
    else if (a == "--calls") callsArg = next();
    else if (a == "--steps") steps = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "--n-train") nTrain = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "--n-test") nTest = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "--wout") wout = true;
    else if (a == "--budget") budgetS = std::atof(next());
    else if (a == "--poll-limit") pollLimit = std::strtoull(next(), nullptr, 0);
    else if (a == "--kernel") kernelPath = next();
    else { std::cerr << "unknown argument " << a << " (see the comment at the top of main.cpp)\n"; return 2; }
  }
  const uint32_t nsh = static_cast<uint32_t>(__builtin_popcountll(shireMask));
  if (runDir.empty() || outDir.empty() || nsh == 0 || shireMask != ((1ull << nsh) - 1) || nTrain > kTrain ||
      nTest > kTest || nTrain == 0 || nTest == 0) {
    std::cerr << "need --run and --out; --shires must be 0x1, 0x3, ... 0xffffffff (shires 0..n-1)\n";
    return 2;
  }
  if (budgetS < 0) budgetS = sysemu ? 1e9 : 8.0;
  if (pollLimit == 0) pollLimit = sysemu ? 2000000ull : 400000000ull;
  const int timeoutS = sysemu ? 3600 : 3;

  // Everything that does not need the device happens before opening it.
  std::vector<std::string> names;
  if (callsArg == "all") {
    names.push_back("warmup");
    for (int i = 0; i < 15; ++i) { std::ostringstream s; s << std::setw(2) << std::setfill('0') << i; names.push_back(s.str()); }
  } else {
    std::istringstream s(callsArg);
    for (std::string n; std::getline(s, n, ',');) names.push_back(n == "warmup" ? n : (n.size() == 1 ? "0" + n : n));
  }
  std::vector<Call> calls(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    calls[i].name = names[i];
    const fs::path p = fs::path(runDir) / (names[i] == "warmup" ? "warmup.bin" : "call_" + names[i] + ".bin");
    if (!loadCall(p, calls[i])) { std::cerr << "cannot read " << p << "\n"; return 1; }
  }
  const auto elf = readFile(kernelPath);
  if (elf.empty()) { std::cerr << "cannot read kernel " << kernelPath << "\n"; return 1; }
  fs::create_directories(outDir);

  std::cout << "mnist: " << calls.size() << " calls, shires=0x" << std::hex << shireMask << std::dec << " ("
            << nsh * MN_ROWS << " rows a step), steps=" << steps << ", n_train=" << nTrain << ", n_test=" << nTest
            << ", device=" << (sysemu ? "sysemu" : "silicon") << "\n";

  struct Timing { std::string name; double in, kernel, out, total; };
  std::vector<Timing> times;
  std::vector<std::vector<MnStats>> stats;
  std::vector<std::vector<int32_t>> labels;
  std::vector<float> wfinal;
  bool ok = true;
  double openMs = 0, heldMs = 0;
  {
    const auto tOpen = Clock::now();
    std::shared_ptr<dev::IDeviceLayer> deviceLayer = makeDeviceLayer(sysemu, simArgs);
    auto runtime = rt::IRuntime::create(deviceLayer);
    const auto devices = runtime->getDevices();
    if (devices.empty()) { std::cerr << "no ET devices found\n"; return 1; }
    const auto device = devices[0];
    const auto stream = runtime->createStream(device);
    const auto load = runtime->loadCode(stream, elf.data(), elf.size());
    runtime->waitForEvent(load.event_);
    const size_t bx = size_t(kTrain) * kDin * 4, by = kTrain * 4, btx = size_t(kTest) * kDin * 4;
    const size_t blab = size_t(kTest) * 64, bstat = MN_MAX_MINIONS * sizeof(MnStats), bw = MN_W_FLOATS * 4;
    auto* dX = runtime->mallocDevice(device, bx);
    auto* dY = runtime->mallocDevice(device, by);
    auto* dTX = runtime->mallocDevice(device, btx);
    auto* dLab = runtime->mallocDevice(device, blab);
    auto* dStat = runtime->mallocDevice(device, bstat);
    auto* dCnt = runtime->mallocDevice(device, 64);
    auto* dW = runtime->mallocDevice(device, bw);
    std::vector<std::byte> zeros(std::max(bstat, blab), std::byte{0});
    openMs = msSince(tOpen);

    for (auto& c : calls) {
      if (msSince(tOpen) > budgetS * 1e3) {
        std::cout << "stopping before call " << c.name << ": device-time budget of " << budgetS << " s used\n";
        break;
      }
      // Untimed: clear the outputs, so a minion that never wrote shows up.
      runtime->memcpyHostToDevice(stream, zeros.data(), dStat, bstat);
      runtime->memcpyHostToDevice(stream, zeros.data(), dLab, blab);
      runtime->waitForStream(stream);

      MnArgs args{};
      args.train_x = reinterpret_cast<uint64_t>(dX);
      args.train_y = reinterpret_cast<uint64_t>(dY);
      args.test_x = reinterpret_cast<uint64_t>(dTX);
      args.labels = reinterpret_cast<uint64_t>(dLab);
      args.wout = wout ? reinterpret_cast<uint64_t>(dW) : 0;
      args.stats = reinterpret_cast<uint64_t>(dStat);
      args.counter = reinterpret_cast<uint64_t>(dCnt);
      args.shire_mask = shireMask;
      args.poll_limit = pollLimit;
      args.n_train = nTrain; args.n_test = nTest; args.steps = steps; args.batch = nsh * MN_ROWS;
      args.lr = 0.4f; args.mu = 0.95f; args.wd = 0.001f; args.clip = 0.03f;
      args.bound1 = std::sqrt(6.0f / 60.0f); args.bound2 = std::sqrt(6.0f / 256.0f);
      args.seed1 = 0x1234567u; args.seed2 = 0x89ABCDEu;

      // Timed: the draw in, the launch, the labels out.
      const auto t0 = Clock::now();
      runtime->memcpyHostToDevice(stream, reinterpret_cast<const std::byte*>(c.x.data()), dX, size_t(nTrain) * kDin * 4);
      runtime->memcpyHostToDevice(stream, reinterpret_cast<const std::byte*>(c.y.data()), dY, size_t(nTrain) * 4);
      runtime->memcpyHostToDevice(stream, reinterpret_cast<const std::byte*>(c.tx.data()), dTX, size_t(nTest) * kDin * 4);
      runtime->memcpyHostToDevice(stream, zeros.data(), dCnt, 64);
      runtime->waitForStream(stream);
      const double inMs = msSince(t0);
      const auto t1 = Clock::now();
      rt::KernelLaunchOptions opts;
      opts.setShireMask(shireMask);
      opts.setBarrier(true);
      runtime->kernelLaunch(stream, load.kernel_, reinterpret_cast<const std::byte*>(&args), sizeof(args), opts);
      ok = runtime->waitForStream(stream, std::chrono::seconds(timeoutS));
      const double kMs = msSince(t1);
      if (!ok) {
        std::cerr << "call " << c.name << ": kernel did not finish within " << timeoutS << " s, aborting the stream\n";
        runtime->waitForEvent(runtime->abortStream(stream));
      }
      for (const auto& e : runtime->retrieveStreamErrors(stream)) {
        std::cerr << "call " << c.name << ": stream error: " << e.getString() << "\n";
        ok = false;
      }
      const auto t2 = Clock::now();
      std::vector<int32_t> lab(size_t(nTest) * 16);
      runtime->memcpyDeviceToHost(stream, dLab, reinterpret_cast<std::byte*>(lab.data()), size_t(nTest) * 64);
      runtime->waitForStream(stream);
      const double outMs = msSince(t2);
      times.push_back({c.name, inMs, kMs, outMs, msSince(t0)});
      std::vector<MnStats> st(MN_MAX_MINIONS);
      runtime->memcpyDeviceToHost(stream, dStat, reinterpret_cast<std::byte*>(st.data()), bstat);
      if (wout && wfinal.empty()) {
        wfinal.resize(MN_W_FLOATS);
        runtime->memcpyDeviceToHost(stream, dW, reinterpret_cast<std::byte*>(wfinal.data()), bw);
      }
      runtime->waitForStream(stream);
      stats.push_back(std::move(st));
      std::vector<int32_t> l(nTest);
      for (uint32_t r = 0; r < nTest; ++r) l[r] = lab[size_t(r) * 16];
      labels.push_back(std::move(l));
      if (!ok) break;
    }
    for (auto* p : {dX, dY, dTX, dLab, dStat, dCnt, dW}) runtime->freeDevice(device, p);
    runtime->unloadCode(load.kernel_);
    runtime->destroyStream(stream);
    heldMs = msSince(tOpen);
  }  // runtime and device layer are destroyed here, releasing the device
  std::cout << std::fixed << std::setprecision(3) << "device open + load " << openMs << " ms, held " << heldMs
            << " ms in all\n";

  // Outputs, with the device closed.
  std::ofstream tj(fs::path(outDir) / "times.json"), sj(fs::path(outDir) / "stats.json");
  tj << "{";
  sj << "{";
  bool first = true;
  int bad = 0;
  for (size_t i = 0; i < times.size(); ++i) {
    const auto& t = times[i];
    const auto& st = stats[i];
    uint32_t err = 0, missing = 0, maxTotal = 0, maxTrain = 0, steps0 = 0;
    std::vector<uint32_t> ph(MN_P_COUNT, 0);
    for (uint32_t m = 0; m < nsh * 32; ++m) {
      const auto& s = st[m];
      if (s.magic != MN_MAGIC) { ++missing; continue; }
      err |= s.error;
      maxTotal = std::max(maxTotal, s.total_cycles);
      maxTrain = std::max(maxTrain, s.train_cycles);
      steps0 = s.steps_done;
      for (int p = 0; p < MN_P_COUNT; ++p) ph[p] = std::max(ph[p], s.phase[p]);
    }
    bad += (missing || err) ? 1 : 0;
    const double mhz = 600.0;
    std::cout << "call " << t.name << ": " << t.total << " ms (in " << t.in << ", kernel " << t.kernel << ", out "
              << t.out << "); device " << maxTotal / mhz / 1e3 << " ms (train " << maxTrain / mhz / 1e3
              << " ms at 600 MHz), steps " << steps0 << (missing ? ", MISSING " + std::to_string(missing) : "")
              << (err ? ", ERROR " + std::to_string(err) : "") << "\n";
    if (t.name != "warmup") {
      tj << (first ? "" : ", ") << "\"" << t.name << "\": " << t.total;
      std::ofstream lf(fs::path(outDir) / ("call_" + t.name + ".labels"), std::ios::binary);
      lf.write(reinterpret_cast<const char*>(labels[i].data()), static_cast<std::streamsize>(labels[i].size() * 4));
      first = false;
    }
    sj << (i ? ", " : "") << "\"" << t.name << "\": {\"in_ms\": " << t.in << ", \"kernel_ms\": " << t.kernel
       << ", \"out_ms\": " << t.out << ", \"total_ms\": " << t.total << ", \"device_cycles\": " << maxTotal
       << ", \"train_cycles\": " << maxTrain << ", \"missing\": " << missing << ", \"error\": " << err
       << ", \"phase_max_cycles\": [";
    for (int p = 0; p < MN_P_COUNT; ++p) sj << (p ? ", " : "") << ph[p];
    sj << "], \"dbg_minion0\": [";
    for (int q = 0; q < 8; ++q) sj << (q ? ", " : "") << st[0].pad[q];
    sj << "], \"dbg_max\": [";
    for (int q = 0; q < 8; ++q) {
      uint32_t m = 0;
      for (uint32_t mm = 0; mm < nsh * 32; ++mm) m = std::max(m, st[mm].pad[q]);
      sj << (q ? ", " : "") << m;
    }
    sj << "]}";
  }
  tj << "}\n";
  sj << "}\n";
  if (!wfinal.empty()) {
    std::ofstream wf(fs::path(outDir) / "wfinal.f32", std::ios::binary);
    wf.write(reinterpret_cast<const char*>(wfinal.data()), static_cast<std::streamsize>(wfinal.size() * 4));
  }
  (void)fbits;
  std::cout << (ok && bad == 0 ? "OK" : "FAIL") << "\n";
  return ok && bad == 0 ? 0 : 1;
}
