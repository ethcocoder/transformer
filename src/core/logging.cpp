#include "aurelis/logging.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace aurelis {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::Logger()
    : current_level_(LogLevel::INFO)
    , console_enabled_(true)
    , file_enabled_(false)
    , max_file_size_(10 * 1024 * 1024)  // 10 MB
    , max_files_(5) {
}

Logger::~Logger() {
    flush();
    if (log_file_) {
        log_file_->close();
    }
}

void Logger::set_log_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_level_ = level;
}

void Logger::set_console_output(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    console_enabled_ = enabled;
}

void Logger::set_file_output(const std::string& filename, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_enabled_ = enabled;
    log_filename_ = filename;
    
    if (file_enabled_) {
        log_file_ = std::make_unique<std::ofstream>(filename, std::ios::out | std::ios::app);
        if (!log_file_->is_open()) {
            std::cerr << "Warning: Could not open log file: " << filename << std::endl;
            file_enabled_ = false;
        }
    } else {
        if (log_file_) {
            log_file_->close();
            log_file_.reset();
        }
    }
}

void Logger::set_max_file_size(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_file_size_ = bytes;
}

void Logger::set_max_files(int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_files_ = std::max(1, count);
}

std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm local_time;
    #ifdef _WIN32
    localtime_s(&local_time, &now_time);
    #else
    localtime_r(&now_time, &local_time);
    #endif
    
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::format_message(LogLevel level, const char* file, int line, const std::string& message) {
    const char* level_str;
    switch (level) {
        case LogLevel::DEBUG: level_str = "DEBUG"; break;
        case LogLevel::INFO: level_str = "INFO"; break;
        case LogLevel::WARNING: level_str = "WARNING"; break;
        case LogLevel::ERROR: level_str = "ERROR"; break;
        case LogLevel::FATAL: level_str = "FATAL"; break;
        default: level_str = "UNKNOWN"; break;
    }
    
    // Extract just the filename without path
    std::string filename = file;
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        filename = filename.substr(last_slash + 1);
    }
    
    std::ostringstream oss;
    oss << "[" << get_timestamp() << "] "
        << "[" << level_str << "] "
        << "[" << filename << ":" << line << "] "
        << message;
    
    return oss.str();
}

void Logger::rotate_logs() {
    // Simple rotation without filesystem for compatibility
    // Just close and reopen to append
    if (log_file_) {
        log_file_->flush();
        log_file_->close();
        log_file_ = std::make_unique<std::ofstream>(log_filename_, std::ios::out | std::ios::app);
    }
}

void Logger::log(LogLevel level, const char* file, int line, const std::string& message) {
    if (level < current_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string formatted = format_message(level, file, line, message);
    
    // Output to console
    if (console_enabled_) {
        std::ostream& out = (level >= LogLevel::WARNING) ? std::cerr : std::cout;
        out << formatted << std::endl;
    }
    
    // Output to file
    if (file_enabled_ && log_file_ && log_file_->is_open()) {
        (*log_file_) << formatted << std::endl;
        log_file_->flush();
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_ && log_file_->is_open()) {
        log_file_->flush();
    }
    std::cout.flush();
    std::cerr.flush();
}

} // namespace aurelis
