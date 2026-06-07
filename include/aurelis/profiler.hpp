#pragma once

#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

namespace aurelis {

class Profiler {
public:
    static Profiler& instance() {
        static Profiler p;
        return p;
    }

    void start(const std::string& name) {
        m_start_times[name] = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto it = m_start_times.find(name);
        if (it != m_start_times.end()) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - it->second).count();
            m_durations[name] += duration;
            m_counts[name] += 1;
        }
    }

    void report() const {
        std::cout << std::setw(30) << "Name"
                  << std::setw(15) << "Total (us)"
                  << std::setw(10) << "Count"
                  << std::setw(15) << "Avg (us)" << "\n";
        for (const auto& kv : m_durations) {
            const std::string& name = kv.first;
            long long total = kv.second;
            long long count = m_counts.at(name);
            double avg = static_cast<double>(total) / static_cast<double>(count);
            std::cout << std::setw(30) << name
                      << std::setw(15) << total
                      << std::setw(10) << count
                      << std::fixed << std::setprecision(2) << std::setw(15) << avg << "\n";
        }
    }

    void reset() {
        m_start_times.clear();
        m_durations.clear();
        m_counts.clear();
    }

private:
    Profiler() = default;

    std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> m_start_times;
    std::map<std::string, long long> m_durations;
    std::map<std::string, long long> m_counts;
};

class ScopedProfiler {
public:
    ScopedProfiler(const std::string& name) : m_name(name) {
        Profiler::instance().start(m_name);
    }
    ~ScopedProfiler() {
        Profiler::instance().stop(m_name);
    }
private:
    std::string m_name;
};

} // namespace aurelis
