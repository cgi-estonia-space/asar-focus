#include <unistd.h>
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

void TimeLog(std::chrono::steady_clock::time_point beg, std::chrono::steady_clock::time_point end, const char* msg) {
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
    std::cout << msg << " time = " << diff << " ms" << std::endl;
}

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

enum class InitializeBufferStrategy {
    ALLOCATE_ONLY,
    ALLOCATE_WRITE_SINGLE_ITEM,
    ALLOCATE_MEMSET_ALL_BUFFER,
    ALLOCATE_WRITE_SINGLE_BYTE_TO_EACH_PAGE,
    ALLOCATE_MMAP
};
char* InitializeBufferFor(size_t bytes, InitializeBufferStrategy strategy) {
    char* buf = new char[bytes];
    if (strategy == InitializeBufferStrategy::ALLOCATE_ONLY) {
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_ITEM) {
        buf[0] = 0x12;
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_MEMSET_ALL_BUFFER) {
        std::memset(buf, 0xAA, BUFFER_OF_1GB.size());
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_BYTE_TO_EACH_PAGE) {
        for (size_t pg{0}; pg < BUFFER_OF_1GB.size() / sysconf(_SC_PAGE_SIZE); pg++) {
            buf[pg * 4096] = 0x12;
        }
        return buf;
    } else if (strategy == InitializeBufferStrategy::ALLOCATE_MMAP) {
        void* ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        return (char*)ptr;
    }

    return nullptr;
}

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
    }

    return "";
}

void RunTest(const char* data_from, size_t size, const InitializeBufferStrategy strategy) {
    const auto strategy_name = std::string(InitializeBufferStrategyStr(strategy));
    const auto the_begin = TimePoint();
    auto* buf = InitializeBufferFor(size, strategy);
    TimeLog(the_begin, TimePoint(), strategy_name.c_str());
    const auto write_start = TimePoint();
    std::memcpy(buf, data_from, size);
    TimeLog(write_start, TimePoint(), "Buffer write time");
    TimeLog(the_begin, TimePoint(), (strategy_name + " total").c_str());
    if (strategy == InitializeBufferStrategy::ALLOCATE_MMAP) {
        munmap(buf, size);
    } else {
        delete[] buf;
    }
}

}  // namespace

int main(int, char*[]) {
    std::cout << "Initializing " << BUFFER_OF_1GB.size() / MiB << "MiB" << std::endl;
    Init();

    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_ONLY);
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_ITEM);
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_MEMSET_ALL_BUFFER);
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(),
            InitializeBufferStrategy::ALLOCATE_WRITE_SINGLE_BYTE_TO_EACH_PAGE);
    RunTest(BUFFER_OF_1GB.data(), BUFFER_OF_1GB.size(), InitializeBufferStrategy::ALLOCATE_MMAP);

    return 0;
}