// S = A * A^T on ET-SoC-1 for a row-major fp32 file A[m][k]; writes S[m][m] row-major fp32.
//   aat_host --in A.f32 -m M -k K --out S.f32 [--sysemu [--sim-args "..."]] [--shires MASK] [--reps R]
//            [--budget S] [--kernel aat.elf] [--dry]
// From workloads/sgemm.  The host builds B = A^T so the kernel keeps sgemm's inner loop; the check against a
// float64 reference is done afterwards by check.py, with the device closed.  --dry does everything except open
// the device.
// On a shared lab card the device is held only between opening the runtime and
// tearing it down; everything else (input generation, the reference product,
// checking) happens with the device closed, and a time budget stops further
// launches if the device has been held too long.
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
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "Constants.h"
#include "aat_args.h"

// libetrt's thread-pool workers log at the custom levels VERBOSE_HIGH/MID/LOW, which only logging::LoggerDefault
// registers with g3log. Left unregistered, the first log call from several new workers inserts the level into g3log's
// level map from all of them at once and can corrupt it: aifoundry3's host crash about 1.08 s into 1 launch in 100
// (docs/findings/14-card-behaviour.md, "Traps"). Registering them, disabled as the map's default would leave them,
// before the runtime starts any thread removes the race.
static void registerRuntimeLogLevels() {
  const LEVELS levels[] = {LEVELS(g3::kDebugValue - 100, "VERBOSE_HIGH"), LEVELS(g3::kDebugValue - 99, "VERBOSE_MID"),
                           LEVELS(g3::kDebugValue - 98, "VERBOSE_LOW")};
  for (const auto& l : levels) g3::only_change_at_initialization::addLogLevel(l, false);
}

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static double secondsSince(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

static std::vector<std::byte> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }
  std::vector<std::byte> data(fs::file_size(path));
  file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return data;
}

static std::unique_ptr<dev::IDeviceLayer> makeDeviceLayer(bool sysemu, const std::string& simArgs) {
  if (!sysemu) {
    return dev::IDeviceLayer::createPcieDeviceLayer(true, true);  // /dev/et0_ops + /dev/et0_mgmt
  }
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
  std::istringstream extra(simArgs);  // e.g. "-vpurf_warn"
  for (std::string arg; extra >> arg;) {
    o.additionalOptions.push_back(arg);
  }
  return dev::IDeviceLayer::createSysEmuDeviceLayer(o, 1);
}

int main(int argc, char** argv) {
  registerRuntimeLogLevels();  // first, before any library starts a thread
  bool sysemu = false;
  uint32_t m = 0, k = 0;
  std::string inPath, outPath;
  bool dry = false;
  uint64_t shireMask = 0xffffffff;
  int reps = 3;
  double budgetS = -1;  // stop launching once the device has been held this long (default: 8 s on silicon)
  std::string kernelPath = KERNEL_ELF;
  std::string simArgs;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::cerr << a << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--sysemu") sysemu = true;
    else if (a == "--dry") dry = true;
    else if (a == "--in") inPath = next();
    else if (a == "--out") outPath = next();
    else if (a == "-m") m = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "-k") k = static_cast<uint32_t>(std::atoi(next()));
    else if (a == "--shires") shireMask = std::strtoull(next(), nullptr, 0);
    else if (a == "--reps") reps = std::max(1, std::atoi(next()));
    else if (a == "--budget") budgetS = std::atof(next());
    else if (a == "--kernel") kernelPath = next();
    else if (a == "--sim-args") simArgs = next();
    else {
      std::cerr << "usage: " << argv[0]
                << " --in A.f32 -m M -k K --out S.f32 [--sysemu] [--sim-args \"-vpurf_warn ...\"] [--shires MASK]"
                   " [--reps R] [--budget SECONDS] [--kernel aat.elf] [--dry]\n";
      return 2;
    }
  }
  if (budgetS < 0) {
    budgetS = sysemu ? 1e9 : 8.0;  // the simulator is private; a real card is shared
  }
  const int timeoutS = sysemu ? 600 : 5;  // per launch; sys_emu steps ~2100 harts serially
  const uint32_t n = m;  // S is m x m
  if (m == 0 || m % 16 != 0 || k == 0 || inPath.empty() || outPath.empty()) {
    std::cerr << "need --in, --out, -m M (a positive multiple of 16) and -k K\n";
    return 2;
  }

  // Everything that does not need the device happens before opening it.
  const auto elf = readFile(kernelPath);
  if (elf.empty()) {
    std::cerr << "cannot read kernel " << kernelPath << "\n";
    return 1;
  }
  const auto raw = readFile(inPath);
  if (raw.size() != size_t(m) * k * sizeof(float)) {
    std::cerr << inPath << " has " << raw.size() << " bytes, want " << size_t(m) * k * sizeof(float) << "\n";
    return 1;
  }
  std::vector<float> A(size_t(m) * k), B(size_t(k) * n), C(size_t(m) * n);
  std::memcpy(A.data(), raw.data(), raw.size());
  for (uint32_t i = 0; i < m; ++i)  // B = A^T, k x n
    for (uint32_t kk = 0; kk < k; ++kk) B[size_t(kk) * n + i] = A[size_t(i) * k + kk];
  std::fill(C.begin(), C.end(), std::numeric_limits<float>::quiet_NaN());  // unwritten -> NaN in the output
  const size_t bytesA = A.size() * sizeof(float), bytesB = B.size() * sizeof(float), bytesC = C.size() * sizeof(float);

  std::cout << "aat m=" << m << " k=" << k << " shires=0x" << std::hex << shireMask << std::dec << " reps=" << reps
            << " device=" << (sysemu ? "sysemu" : "silicon") << (dry ? " DRY" : "") << "\n";
  if (dry) {
    std::cout << "dry run: device not opened\n";
    return 0;
  }

  std::vector<double> launchS;
  double heldS = 0;
  {
    const auto tOpen = Clock::now();
    std::shared_ptr<dev::IDeviceLayer> deviceLayer = makeDeviceLayer(sysemu, simArgs);
    auto runtime = rt::IRuntime::create(deviceLayer);
    const auto devices = runtime->getDevices();
    if (devices.empty()) {
      std::cerr << "no ET devices found\n";
      return 1;
    }
    const auto device = devices[0];
    const auto stream = runtime->createStream(device);
    const double openS = secondsSince(tOpen);

    const auto load = runtime->loadCode(stream, elf.data(), elf.size());
    runtime->waitForEvent(load.event_);
    auto* dA = runtime->mallocDevice(device, bytesA);
    auto* dB = runtime->mallocDevice(device, bytesB);
    auto* dC = runtime->mallocDevice(device, bytesC);
    runtime->memcpyHostToDevice(stream, reinterpret_cast<std::byte*>(A.data()), dA, bytesA);
    runtime->memcpyHostToDevice(stream, reinterpret_cast<std::byte*>(B.data()), dB, bytesB);
    runtime->memcpyHostToDevice(stream, reinterpret_cast<std::byte*>(C.data()), dC, bytesC);
    runtime->waitForStream(stream);
    const double setupS = secondsSince(tOpen) - openS;

    AatArgs args{reinterpret_cast<uint64_t>(dA), reinterpret_cast<uint64_t>(dB),
                 reinterpret_cast<uint64_t>(dC), shireMask, m, n, k, 0};
    rt::KernelLaunchOptions opts;
    opts.setShireMask(shireMask);
    opts.setBarrier(true);
    bool ok = true;
    for (int r = 0; r < reps && ok; ++r) {
      if (secondsSince(tOpen) > budgetS) {
        std::cout << "stopping after " << r << " launches: device-time budget of " << budgetS << " s used\n";
        break;
      }
      const auto t0 = Clock::now();
      runtime->kernelLaunch(stream, load.kernel_, reinterpret_cast<const std::byte*>(&args), sizeof(args), opts);
      ok = runtime->waitForStream(stream, std::chrono::seconds(timeoutS));
      launchS.push_back(secondsSince(t0));
      if (!ok) {
        std::cerr << "kernel did not finish within " << timeoutS << " s, aborting the stream\n";
        runtime->waitForEvent(runtime->abortStream(stream));
      }
      for (const auto& e : runtime->retrieveStreamErrors(stream)) {
        std::cerr << "stream error: " << e.getString() << "\n";
        ok = false;
      }
    }
    const auto tOut = Clock::now();
    runtime->memcpyDeviceToHost(stream, dC, reinterpret_cast<std::byte*>(C.data()), bytesC);
    runtime->waitForStream(stream);
    const double copyOutS = secondsSince(tOut);
    runtime->freeDevice(device, dA);
    runtime->freeDevice(device, dB);
    runtime->freeDevice(device, dC);
    runtime->unloadCode(load.kernel_);
    runtime->destroyStream(stream);
    heldS = secondsSince(tOpen);
    std::cout << "device open " << openS << " s, load+copy-in " << setupS << " s, copy-out " << copyOutS << " s\n";
    if (!ok) {
      std::cout << "FAIL (kernel error)\n";
      return 1;
    }
  }  // runtime and device layer are destroyed here, releasing the device
  std::cout << "device held for " << heldS << " s total\n";

  std::ofstream out(outPath, std::ios::binary);
  out.write(reinterpret_cast<const char*>(C.data()), static_cast<std::streamsize>(bytesC));
  if (!out) {
    std::cerr << "cannot write " << outPath << "\n";
    return 1;
  }
  const size_t nonFinite = static_cast<size_t>(std::count_if(C.begin(), C.end(), [](float x) { return !std::isfinite(x); }));
  const double flop = 2.0 * m * double(n) * k;
  for (size_t r = 0; r < launchS.size(); ++r) {
    std::cout << "launch " << r << ": " << launchS[r] * 1e3 << " ms, " << flop / launchS[r] * 1e-9 << " GFLOP/s\n";
  }
  std::cout << "wrote " << outPath << " (" << bytesC << " bytes), non-finite entries " << nonFinite << "\n";
  return nonFinite == 0 && !launchS.empty() ? 0 : 1;
}
