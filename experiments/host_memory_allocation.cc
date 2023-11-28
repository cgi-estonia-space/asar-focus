#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <random>
#include <string_view>

#include <sys/mman.h>

namespace {
constexpr int MiB = 1 << 20;  // 1 MB = 2^20 bytes

std::array<char, 1000 * MiB> BUFFER_OF_1GB{};

auto TimePoint() { return std::chrono::steady_clock::now(); }

void Init() {
    // Use random_device to obtain seed for the random number engine
    std::random_device rd;
    // Use Mersenne Twister 19937 as the random number engine
    std::mt19937 gen(rd());
    // Define the distribution for random values (you can adjust the range as needed)
    std::uniform_int_distribution<char> dis(CHAR_MIN, CHAR_MAX);

    // Fill the array with random values
    for (auto& e : BUFFER_OF_1GB) {
        e = dis(gen);
    }
}

struct Benchmark {
    std::string name;
    uint32_t init_time_ms;
    uint32_t write_time_ms;
    uint32_t total_time_ms;
};

enum class InitializeBufferStrategy {
    ALLOCATE_ONLY,
    ALLOCATE_WRITE_SINGLE_ITEM,
    ALLOCATE_MEMSET_ALL_BUFFER,
    ALLOCATE_WRITE_SINGLE_BYTE_TO_EACH_PAGE,
    ALLOCATE_MMAP,
    ALLOCATE_READ,
    ALLOCATE_MADVISE
};

std::string_view InitializeBufferStrategyStr(const InitializeBufferStrategy strategy) {
    switch (strategy) {
        case InitializeBufferStrategy::ALLOCATE_ONLY:
            return "ALLOCATE_ONLY";
        case InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_ITEM:
            return "ALLOCATE_WRITE_SINGLE_ITEM";
        case InitializeBufferStrategy::ALLOCATE_MEMSET_ALL_BUFFER:
            return "ALLOCATE_MEMSET_ALL_BUFFER";
        case InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_BYTE_TO_EACH_PAGE:
            return "ALLOCATE_WRITE_SINGLE_BYTE_TO_EACH_PAGE";
        case InitializeBufferStrategy::ALLOCATE_MMAP:
            return "ALLOCATE_MMAP";
        case InitializeBufferStrategy::ALLOCATE_READ:
            return "ALLOCATE_READ";
        case InitializeBufferStrategy::ALLOCATE_MADVISE:
            return "ALLOCATE_MADVISE";
    }

    return "";
}

char* InitializeBufferFor(size_t bytes, InitializeBufferStrategy strategy) {
    if (strategy == InitializeBufferStrategy::ALLOCATE_ONLY) {
        char* buf = new char[bytes];
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_ITEM) {
        char* buf = new char[bytes];
        buf[0] = 0x12;
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_MEMSET_ALL_BUFFER) {
        char* buf = new char[bytes];
        std::memset(buf, 0xAA, BUFFER_OF_1GB.size());
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_BYTE_TO_EACH_PAGE) {
        char* buf = new char[bytes];
        const auto PAGE_SIZE = sysconf(_SC_PAGE_SIZE);
        for (size_t pg{0}; pg < BUFFER_OF_1GB.size() / PAGE_SIZE; pg++) {
            buf[pg * 4096] = 0x12;
        }
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_MMAP) {
        void* ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        return (char*)ptr;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_READ) {
        char* buf = new char[bytes];
        const auto PAGE_SIZE = sysconf(_SC_PAGE_SIZE);
        for (size_t i{0}; i < bytes; i += PAGE_SIZE) {
            volatile char* tere = buf;
            [[maybe_unused]] char tmp = tere[i];
        }
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_MADVISE) {
        errno = 0;
        void* ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        const auto res = madvise(ptr, bytes, MADV_WILLNEED | MADV_POPULATE_READ);
        if (res != 0) {
            std::cerr << InitializeBufferStrategyStr(strategy) << " failed" << std::endl;
            std::cerr << strerror(errno) << std::endl;
        }
        return (char*)ptr;
    }

    return nullptr;
}

void RunTest(const char* data_from, size_t size, const InitializeBufferStrategy strategy, Benchmark& benchmark) {
    const auto strategy_name = std::string(InitializeBufferStrategyStr(strategy));
    benchmark.name = strategy_name;
    const auto the_begin = TimePoint();
    auto* buf = InitializeBufferFor(size, strategy);
    benchmark.init_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(TimePoint() - the_begin).count();
    const auto write_start = TimePoint();
    std::memcpy(buf, data_from, size);
    benchmark.write_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(TimePoint() - write_start).count();
    benchmark.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(TimePoint() - the_begin).count();
    if (strategy == InitializeBufferStrategy::ALLOCATE_MMAP || strategy == InitializeBufferStrategy::ALLOCATE_MADVISE) {
        munmap(buf, size);
    } else {
        delete[] buf;
    }
}

void PrintInits(const std::vector<Benchmark>& b, size_t name_width) {
    auto cb = b;
    std::sort(cb.begin(), cb.end(),
              [](const Benchmark& a, const Benchmark& b) { return a.init_time_ms < b.init_time_ms; });

    std::cout << "Init times (buffer creation) ms" << std::endl;
    for (const auto& c : cb) {
        std::cout << c.name;
        for (size_t i = c.name.length(); i < name_width + 1; i++) {
            std::cout << " ";
        }
        std::cout << c.init_time_ms << std::endl;
    }
}

void PrintWrites(const std::vector<Benchmark>& b, size_t name_width) {
    auto cb = b;
    std::sort(cb.begin(), cb.end(),
              [](const Benchmark& a, const Benchmark& b) { return a.write_time_ms < b.write_time_ms; });

    std::cout << "Write times ms" << std::endl;
    for (const auto& c : cb) {
        std::cout << c.name;
        for (size_t i = c.name.length(); i < name_width + 1; i++) {
            std::cout << " ";
        }
        std::cout << c.write_time_ms << std::endl;
    }
}

void PrintTotals(const std::vector<Benchmark>& b, size_t name_width) {
    auto cb = b;
    std::sort(cb.begin(), cb.end(),
              [](const Benchmark& a, const Benchmark& b) { return a.total_time_ms < b.total_time_ms; });

    std::cout << "Total times ms" << std::endl;
    for (const auto& c : cb) {
        std::cout << c.name;
        for (size_t i = c.name.length(); i < name_width + 1; i++) {
            std::cout << " ";
        }
        std::cout << c.total_time_ms << std::endl;
    }
}

}  // namespace

int main(int, char*[]) {
    std::cout << "Initializing " << BUFFER_OF_1GB.size() / MiB << "MiB" << std::endl;
    Init();

    std::vector<Benchmark> benchmarks(7);
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_ONLY, benchmarks.at(0));
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_ITEM,
            benchmarks.at(1));
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_MEMSET_ALL_BUFFER,
            benchmarks.at(2));
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(),
            InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_BYTE_TO_EACH_PAGE, benchmarks.at(3));
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_MMAP, benchmarks.at(4));
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_READ, benchmarks.at(5));
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_MADVISE, benchmarks.at(6));

    size_t max_length = 0;
    // Find the longest string in the vector
    for (const auto& benchmark : benchmarks) {
        if (benchmark.name.length() > max_length) {
            max_length = benchmark.name.length();
        }
    }
    PrintInits(benchmarks, max_length);
    std::cout << std::endl;
    PrintWrites(benchmarks, max_length);
    std::cout << std::endl;
    PrintTotals(benchmarks, max_length);

    return 0;
}