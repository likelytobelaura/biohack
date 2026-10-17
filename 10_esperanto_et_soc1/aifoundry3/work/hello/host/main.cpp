// hello on ET-SoC-1: every hart of the launched shires writes a 64-byte record about itself; the host checks
// that each one reported in with the right identity, and writes every record to a CSV.
//   hello_host [--sysemu [--sim-args "..."]] [--shires MASK] [--reps R] [--csv FILE] [--kernel hello.elf]
// Host structure copied from ~/et-soc1-prototyping/workloads/sgemm/host/main.cpp: the device is held only
// between opening the runtime and tearing it down, and a time budget stops further launches.
#include <g3log/loglevels.hpp>
#include <runtime/IRuntime.h>
#include <runtime/Types.h>
#include <device-layer/IDeviceLayer.h>
#include <sw-sysemu/SysEmuOptions.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "Constants.h"
#include "hello_args.h"

// The runtime's logging race (docs/findings/14-card-behaviour.md, "Traps"): register its levels before any thread.
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
  std::istringstream extra(simArgs);
  for (std::string arg; extra >> arg;) {
    o.additionalOptions.push_back(arg);
  }
  return dev::IDeviceLayer::createSysEmuDeviceLayer(o, 1);
}

constexpr uint32_t kHarts = 32 * 64;
constexpr uint32_t kUnwritten = 0xFFFFFFFFu;

int main(int argc, char** argv) {
  registerRuntimeLogLevels();  // first, before any library starts a thread
  bool sysemu = false;
  uint64_t shireMask = 0xffffffff;
  int reps = 3;
  double budgetS = -1;
  std::string kernelPath = KERNEL_ELF;
  std::string simArgs;
  std::string csvPath = "records.csv";
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
    else if (a == "--shires") shireMask = std::strtoull(next(), nullptr, 0) & 0xffffffffull;
    else if (a == "--reps") reps = std::max(1, std::atoi(next()));
    else if (a == "--budget") budgetS = std::atof(next());
    else if (a == "--kernel") kernelPath = next();
    else if (a == "--sim-args") simArgs = next();
    else if (a == "--csv") csvPath = next();
    else {
      std::cerr << "usage: " << argv[0]
                << " [--sysemu] [--sim-args \"...\"] [--shires MASK] [--reps R] [--budget SECONDS] [--csv FILE]"
                   " [--kernel hello.elf]\n";
      return 2;
    }
  }
  if (budgetS < 0) {
    budgetS = sysemu ? 1e9 : 8.0;
  }
  const int timeoutS = sysemu ? 600 : 5;
  if (shireMask == 0) {
    std::cerr << "--shires must name at least one of shires 0-31\n";
    return 2;
  }

  const auto elf = readFile(kernelPath);
  if (elf.empty()) {
    std::cerr << "cannot read kernel " << kernelPath << "\n";
    return 1;
  }
  const size_t words = size_t(kHarts) * HELLO_WORDS;
  const size_t bytes = words * sizeof(uint32_t);
  const std::vector<uint32_t> blank(words, kUnwritten);
  std::vector<std::vector<uint32_t>> results;  // one copy of every record per launch
  std::vector<double> launchS;

  std::cout << "hello shires=0x" << std::hex << shireMask << std::dec << " reps=" << reps
            << " device=" << (sysemu ? "sysemu" : "silicon") << "\n";

  double openS = 0, loadS = 0, heldS = 0;
  bool ok = true;
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
    openS = secondsSince(tOpen);

    const auto load = runtime->loadCode(stream, elf.data(), elf.size());
    runtime->waitForEvent(load.event_);
    auto* dOut = runtime->mallocDevice(device, bytes);
    loadS = secondsSince(tOpen) - openS;

    rt::KernelLaunchOptions opts;
    opts.setShireMask(shireMask);
    opts.setBarrier(true);
    for (int r = 0; r < reps && ok; ++r) {
      if (secondsSince(tOpen) > budgetS) {
        std::cout << "stopping after " << r << " launches: device-time budget of " << budgetS << " s used\n";
        break;
      }
      // Fill the output with a pattern no hart writes, so a hart that never ran shows up.
      runtime->memcpyHostToDevice(stream, reinterpret_cast<const std::byte*>(blank.data()), dOut, bytes);
      runtime->waitForStream(stream);
      HelloArgs args{reinterpret_cast<uint64_t>(dOut), shireMask, static_cast<uint32_t>(r), 0};
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
      std::vector<uint32_t> out(words);
      runtime->memcpyDeviceToHost(stream, dOut, reinterpret_cast<std::byte*>(out.data()), bytes);
      runtime->waitForStream(stream);
      results.push_back(std::move(out));
    }
    runtime->freeDevice(device, dOut);
    runtime->unloadCode(load.kernel_);
    runtime->destroyStream(stream);
    heldS = secondsSince(tOpen);
  }  // runtime and device layer are destroyed here, releasing the device
  std::cout << "device open " << openS * 1e3 << " ms, kernel load + alloc " << loadS * 1e3 << " ms, held "
            << heldS * 1e3 << " ms in all\n";
  if (!ok) {
    std::cout << "FAIL (kernel error)\n";
    return 1;
  }

  std::ofstream csv(csvPath);
  csv << "launch,hart,shire,neighbourhood,minion,thread,magic_ok,loop_cycles,loop_result\n";
  uint32_t expected = 0;
  for (uint32_t h = 0; h < kHarts; ++h) expected += (shireMask >> (h / 64)) & 1;
  bool allPass = !results.empty();
  for (size_t r = 0; r < results.size(); ++r) {
    const auto& out = results[r];
    uint32_t reported = 0, wrong = 0, stray = 0;
    std::vector<uint32_t> cyc;
    for (uint32_t h = 0; h < kHarts; ++h) {
      const uint32_t* rec = &out[size_t(h) * HELLO_WORDS];
      const bool launched = (shireMask >> (h / 64)) & 1;
      const bool wrote = rec[0] != kUnwritten;
      if (!launched) {
        stray += wrote;
        continue;
      }
      if (!wrote) continue;
      ++reported;
      const uint32_t minion = (h >> 1) & 31, thread = h & 1;
      const bool good = rec[0] == HELLO_MAGIC && rec[1] == h && rec[2] == h / 64 && rec[3] == minion &&
                        rec[4] == thread && rec[5] == minion / 8 && rec[6] == r &&
                        (thread == 1 || rec[8] == HELLO_LOOP);
      if (!good) {
        if (wrong < 5) std::cerr << "launch " << r << ": hart " << h << " wrote a bad record\n";
        ++wrong;
      }
      if (thread == 0) cyc.push_back(rec[7]);
      csv << r << "," << h << "," << rec[2] << "," << rec[5] << "," << rec[3] << "," << rec[4] << ","
          << (rec[0] == HELLO_MAGIC) << "," << rec[7] << "," << rec[8] << "\n";
    }
    std::sort(cyc.begin(), cyc.end());
    const bool pass = reported == expected && wrong == 0 && stray == 0;
    allPass = allPass && pass;
    std::cout << "launch " << r << ": " << launchS[r] * 1e3 << " ms, " << reported << "/" << expected
              << " harts reported in from " << __builtin_popcountll(shireMask) << " shires, " << wrong
              << " bad records, " << stray << " stray";
    if (!cyc.empty()) {
      std::cout << "; " << HELLO_LOOP << " dependent adds took " << cyc.front() << " / " << cyc[cyc.size() / 2]
                << " / " << cyc.back() << " cycles (min / median / max over " << cyc.size() << " minions)";
    }
    std::cout << (pass ? "  PASS" : "  FAIL") << "\n";
  }
  std::cout << "records written to " << csvPath << "\n";
  std::cout << (allPass ? "PASS" : "FAIL") << "\n";
  return allPass ? 0 : 1;
}
