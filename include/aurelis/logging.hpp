#pragma once

#include <cstdint>
#include <string>

namespace aurelis {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

void set_log_level(LogLevel level);

void log(LogLevel level, const char* file, int line, const std::string& message);

#define LOG_DEBUG(msg) log(LogLevel::DEBUG, __FILE__, __LINE__, msg)
#define LOG_INFO(msg) log(LogLevel::INFO, __FILE__, __LINE__, msg)
#define LOG_WARNING(msg) log(LogLevel::WARNING, __FILE__, __LINE__, msg)
#define LOG_ERROR(msg) log(LogLevel::ERROR, __FILE__, __LINE__, msg)

}  // namespace aurelis
