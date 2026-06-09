#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <fstream>
#include <mutex>
#include <vector>

namespace aurelis {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

class Logger {
public:
    static Logger& instance();
    
    void set_log_level(LogLevel level);
    void set_console_output(bool enabled);
    void set_file_output(const std::string& filename, bool enabled = true);
    void set_max_file_size(std::size_t bytes);
    void set_max_files(int count);
    
    void log(LogLevel level, const char* file, int line, const std::string& message);
    
    void flush();
    
private:
    Logger();
    ~Logger();
    
    void rotate_logs();
    std::string get_timestamp();
    std::string format_message(LogLevel level, const char* file, int line, const std::string& message);
    
    LogLevel current_level_;
    bool console_enabled_;
    bool file_enabled_;
    std::string log_filename_;
    std::unique_ptr<std::ofstream> log_file_;
    std::size_t max_file_size_;
    int max_files_;
    std::mutex mutex_;
};

// Convenience macros
#define LOG_DEBUG(msg) aurelis::Logger::instance().log(aurelis::LogLevel::DEBUG, __FILE__, __LINE__, msg)
#define LOG_INFO(msg) aurelis::Logger::instance().log(aurelis::LogLevel::INFO, __FILE__, __LINE__, msg)
#define LOG_WARNING(msg) aurelis::Logger::instance().log(aurelis::LogLevel::WARNING, __FILE__, __LINE__, msg)
#define LOG_ERROR(msg) aurelis::Logger::instance().log(aurelis::LogLevel::ERROR, __FILE__, __LINE__, msg)
#define LOG_FATAL(msg) aurelis::Logger::instance().log(aurelis::LogLevel::FATAL, __FILE__, __LINE__, msg)

// Backward compatibility
inline void set_log_level(LogLevel level) {
    Logger::instance().set_log_level(level);
}

inline void log(LogLevel level, const char* file, int line, const std::string& message) {
    Logger::instance().log(level, file, line, message);
}

} // namespace aurelis