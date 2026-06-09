#pragma once

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <sstream>

namespace aurelis {

struct ProfilerStats {
    std::string name;
    long long total_time_us;
    long long count;
    double avg_time_us;
    long long min_time_us;
    long long max_time_us;
};

class Profiler {
public:
    static Profiler& instance();
    
    void start(const std::string& name);
    void stop(const std::string& name);
    void reset();
    
    // Reporting methods
    std::string report_text();
    std::string report_json();
    void report_to_file(const std::string& filename, const std::string& format = "text");
    void print_report();
    
    // Get individual stats
    ProfilerStats get_stats(const std::string& name);
    std::vector<ProfilerStats> get_all_stats();
    
    // Enable/disable profiling
    void set_enabled(bool enabled);
    bool is_enabled() const;
    
private:
    Profiler();
    
    struct TimeRecord {
        std::chrono::high_resolution_clock::time_point start_time;
        long long total_time_us;
        long long count;
        long long min_time_us;
        long long max_time_us;
    };
    
    std::map<std::string, TimeRecord> records_;
    std::mutex mutex_;
    bool enabled_;
};

class ScopedProfiler {
public:
    explicit ScopedProfiler(const std::string& name);
    ~ScopedProfiler();
    
    // Disable copy and move
    ScopedProfiler(const ScopedProfiler&) = delete;
    ScopedProfiler& operator=(const ScopedProfiler&) = delete;
    ScopedProfiler(ScopedProfiler&&) = delete;
    ScopedProfiler& operator=(ScopedProfiler&&) = delete;
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace aurelis
