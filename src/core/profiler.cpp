#include "aurelis/profiler.hpp"
#include <limits>
#include <iomanip>

namespace aurelis {

Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

Profiler::Profiler()
    : enabled_(true) {
}

void Profiler::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

bool Profiler::is_enabled() const {
    return enabled_;
}

void Profiler::start(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return;
    
    auto it = records_.find(name);
    if (it == records_.end()) {
        TimeRecord record;
        record.start_time = std::chrono::high_resolution_clock::now();
        record.total_time_us = 0;
        record.count = 0;
        record.min_time_us = std::numeric_limits<long long>::max();
        record.max_time_us = 0;
        records_[name] = record;
    } else {
        it->second.start_time = std::chrono::high_resolution_clock::now();
    }
}

void Profiler::stop(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return;
    
    auto it = records_.find(name);
    if (it == records_.end()) {
        return;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - it->second.start_time).count());
    
    it->second.total_time_us += duration;
    it->second.count++;
    it->second.min_time_us = std::min(it->second.min_time_us, duration);
    it->second.max_time_us = std::max(it->second.max_time_us, duration);
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
}

ProfilerStats Profiler::get_stats(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    ProfilerStats stats;
    stats.name = name;
    stats.total_time_us = 0;
    stats.count = 0;
    stats.avg_time_us = 0.0;
    stats.min_time_us = 0;
    stats.max_time_us = 0;
    
    auto it = records_.find(name);
    if (it == records_.end()) {
        return stats;
    }
    
    stats.total_time_us = it->second.total_time_us;
    stats.count = it->second.count;
    if (stats.count > 0) {
        stats.avg_time_us = static_cast<double>(stats.total_time_us) / static_cast<double>(stats.count);
    }
    stats.min_time_us = it->second.min_time_us == std::numeric_limits<long long>::max() ? 0 : it->second.min_time_us;
    stats.max_time_us = it->second.max_time_us;
    
    return stats;
}

std::vector<ProfilerStats> Profiler::get_all_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ProfilerStats> result;
    
    for (const auto& pair : records_) {
        ProfilerStats stats;
        stats.name = pair.first;
        stats.total_time_us = pair.second.total_time_us;
        stats.count = pair.second.count;
        if (stats.count > 0) {
            stats.avg_time_us = static_cast<double>(stats.total_time_us) / static_cast<double>(stats.count);
        } else {
            stats.avg_time_us = 0.0;
        }
        stats.min_time_us = pair.second.min_time_us == std::numeric_limits<long long>::max() ? 0 : pair.second.min_time_us;
        stats.max_time_us = pair.second.max_time_us;
        result.push_back(stats);
    }
    
    // Sort by total time descending
    std::sort(result.begin(), result.end(), [](const ProfilerStats& a, const ProfilerStats& b) {
        return a.total_time_us > b.total_time_us;
    });
    
    return result;
}

std::string Profiler::report_text() {
    auto stats = get_all_stats();
    std::ostringstream oss;
    
    oss << std::string(80, '=') << "\n";
    oss << "                          PROFILER REPORT\n";
    oss << std::string(80, '=') << "\n";
    oss << std::left
        << std::setw(30) << "Name"
        << std::setw(15) << "Total (us)"
        << std::setw(10) << "Count"
        << std::setw(15) << "Avg (us)"
        << std::setw(12) << "Min (us)"
        << std::setw(12) << "Max (us)"
        << "\n";
    oss << std::string(80, '-') << "\n";
    
    for (const auto& stat : stats) {
        oss << std::left
            << std::setw(30) << stat.name
            << std::setw(15) << stat.total_time_us
            << std::setw(10) << stat.count
            << std::fixed << std::setprecision(2) << std::setw(15) << stat.avg_time_us
            << std::setw(12) << stat.min_time_us
            << std::setw(12) << stat.max_time_us
            << "\n";
    }
    
    oss << std::string(80, '=') << "\n";
    
    return oss.str();
}

std::string Profiler::report_json() {
    auto stats = get_all_stats();
    std::ostringstream oss;
    
    oss << "[\n";
    bool first = true;
    for (const auto& stat : stats) {
        if (!first) oss << ",\n";
        first = false;
        oss << "  {\n"
            << "    \"name\": \"" << stat.name << "\",\n"
            << "    \"total_time_us\": " << stat.total_time_us << ",\n"
            << "    \"count\": " << stat.count << ",\n"
            << "    \"avg_time_us\": " << std::fixed << std::setprecision(2) << stat.avg_time_us << ",\n"
            << "    \"min_time_us\": " << stat.min_time_us << ",\n"
            << "    \"max_time_us\": " << stat.max_time_us << "\n"
            << "  }";
    }
    oss << "\n]";
    
    return oss.str();
}

void Profiler::report_to_file(const std::string& filename, const std::string& format) {
    std::ofstream ofs(filename);
    if (!ofs) {
        return;
    }
    
    if (format == "json") {
        ofs << report_json();
    } else {
        ofs << report_text();
    }
}

void Profiler::print_report() {
    std::cout << report_text();
}

// ScopedProfiler implementation
ScopedProfiler::ScopedProfiler(const std::string& name)
    : name_(name) {
    if (Profiler::instance().is_enabled()) {
        start_time_ = std::chrono::high_resolution_clock::now();
        Profiler::instance().start(name);
    }
}

ScopedProfiler::~ScopedProfiler() {
    if (Profiler::instance().is_enabled()) {
        Profiler::instance().stop(name_);
    }
}

} // namespace aurelis
