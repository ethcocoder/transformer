#include "aurelis/logging.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <mutex>

namespace aurelis {

namespace {

LogLevel current_log_level = LogLevel::INFO;
std::mutex log_mutex;

const char* level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time));
    return std::string(buffer);
}

}  // namespace

void set_log_level(LogLevel level) {
    current_log_level = level;
}

void log(LogLevel level, const char* file, int line, const std::string& message) {
    if (level < current_log_level) {
        return;
    }

    std::lock_guard<std::mutex> lock(log_mutex);

    std::cerr << "[" << get_timestamp() << "] [" << level_to_string(level) << "] [" << file << ":" << line << "] " << message << std::endl;
}

}  // namespace aurelis
