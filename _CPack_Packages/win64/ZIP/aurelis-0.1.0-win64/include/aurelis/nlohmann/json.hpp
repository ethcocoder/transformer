#pragma once

#include <cctype>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nlohmann {

class parse_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class type_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class json {
public:
    enum class value_t {
        null, object, array, string, boolean, number_integer, number_unsigned, number_float
    };

    json() : type_(value_t::null) {}
    json(std::nullptr_t) : type_(value_t::null) {}
    json(bool val) : type_(value_t::boolean), bool_val_(val) {}
    json(int val) : type_(value_t::number_integer), int_val_(val) {}
    json(long val) : type_(value_t::number_integer), int_val_(val) {}
    json(long long val) : type_(value_t::number_integer), int_val_(val) {}
    json(unsigned int val) : type_(value_t::number_unsigned), uint_val_(val) {}
    json(unsigned long val) : type_(value_t::number_unsigned), uint_val_(val) {}
    json(unsigned long long val) : type_(value_t::number_unsigned), uint_val_(val) {}
    json(double val) : type_(value_t::number_float), double_val_(val) {}
    json(float val) : type_(value_t::number_float), double_val_(val) {}
    json(const std::string& val) : type_(value_t::string), string_val_(val) {}
    json(const char* val) : type_(value_t::string), string_val_(val) {}

    json& operator[](const std::string& key) {
        if (type_ != value_t::object) {
            type_ = value_t::object;
            object_val_.clear();
        }
        return object_val_[key];
    }

    const json& operator[](const std::string& key) const {
        if (type_ != value_t::object) {
            throw std::runtime_error("not an object");
        }
        return object_val_.at(key);
    }

    bool contains(const std::string& key) const {
        return type_ == value_t::object && object_val_.count(key) > 0;
    }

    template<typename T>
    T value(const std::string& key, T default_val) const {
        if (contains(key)) {
            return (*this)[key].get<T>();
        }
        return default_val;
    }

    template<typename T>
    T get() const {
        if constexpr (std::is_same_v<T, bool>) {
            if (type_ != value_t::boolean) throw std::runtime_error("not a boolean");
            return bool_val_;
        } else if constexpr (std::is_integral_v<T>) {
            if (type_ == value_t::number_integer) return static_cast<T>(int_val_);
            if (type_ == value_t::number_unsigned) return static_cast<T>(uint_val_);
            throw std::runtime_error("not a number");
        } else if constexpr (std::is_floating_point_v<T>) {
            if (type_ == value_t::number_float) return static_cast<T>(double_val_);
            if (type_ == value_t::number_integer) return static_cast<T>(int_val_);
            if (type_ == value_t::number_unsigned) return static_cast<T>(uint_val_);
            throw std::runtime_error("not a number");
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (type_ != value_t::string) throw std::runtime_error("not a string");
            return string_val_;
        }
        throw std::runtime_error("unsupported type");
    }

    std::string dump(int indent = -1) const {
        std::ostringstream oss;
        dump_impl(oss, 0, indent);
        return oss.str();
    }

    static json parse(const std::string& s) {
        return parse(s.c_str(), s.c_str() + s.size());
    }

    static json parse(std::istream& is) {
        std::string s((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
        return parse(s);
    }

    value_t type() const { return type_; }
    bool is_object() const { return type_ == value_t::object; }

private:
    value_t type_;
    bool bool_val_ = false;
    long long int_val_ = 0;
    unsigned long long uint_val_ = 0;
    double double_val_ = 0.0;
    std::string string_val_;
    std::map<std::string, json> object_val_;
    std::vector<json> array_val_;

    void dump_impl(std::ostringstream& oss, int depth, int indent) const {
        bool pretty = indent >= 0;
        std::string ind;
        bool first = true;
        if (pretty) ind = std::string(depth * indent, ' ');

        switch (type_) {
            case value_t::null:
                oss << "null";
                break;
            case value_t::boolean:
                oss << (bool_val_ ? "true" : "false");
                break;
            case value_t::number_integer:
                oss << int_val_;
                break;
            case value_t::number_unsigned:
                oss << uint_val_;
                break;
            case value_t::number_float:
                oss << double_val_;
                break;
            case value_t::string:
                oss << "\"";
                for (char c : string_val_) {
                    if (c == '"') oss << "\\\"";
                    else if (c == '\\') oss << "\\\\";
                    else if (c == '\n') oss << "\\n";
                    else if (c == '\r') oss << "\\r";
                    else if (c == '\t') oss << "\\t";
                    else oss << c;
                }
                oss << "\"";
                break;
            case value_t::object:
                oss << "{";
                if (pretty) oss << "\n";
                first = true;
                for (const auto& kv : object_val_) {
                    if (!first) oss << ",";
                    if (pretty) oss << "\n" << std::string((depth + 1) * indent, ' ');
                    first = false;
                    json key_json(kv.first);
                    key_json.dump_impl(oss, depth + 1, indent);
                    oss << ": ";
                    kv.second.dump_impl(oss, depth + 1, indent);
                }
                if (pretty) oss << "\n" << ind;
                oss << "}";
                break;
            case value_t::array:
                oss << "[";
                if (pretty) oss << "\n";
                first = true;
                for (const auto& item : array_val_) {
                    if (!first) oss << ",";
                    if (pretty) oss << "\n" << std::string((depth + 1) * indent, ' ');
                    first = false;
                    item.dump_impl(oss, depth + 1, indent);
                }
                if (pretty) oss << "\n" << ind;
                oss << "]";
                break;
        }
    }

    static json parse_impl(const char*& p, const char* end) {
        while (p < end && std::isspace(*p)) p++;
        if (p >= end) throw std::runtime_error("unexpected end");
        
        if (*p == '{') {
            p++;
            json obj;
            obj.type_ = value_t::object;
            bool first = true;
            while (true) {
                while (p < end && std::isspace(*p)) p++;
                if (p >= end) throw std::runtime_error("unexpected end");
                if (*p == '}') {
                    p++;
                    break;
                }
                if (!first) {
                    if (*p != ',') throw std::runtime_error("expected comma");
                    p++;
                    while (p < end && std::isspace(*p)) p++;
                }
                first = false;
                json key = parse_impl(p, end);
                while (p < end && std::isspace(*p)) p++;
                if (*p != ':') throw std::runtime_error("expected colon");
                p++;
                json val = parse_impl(p, end);
                obj.object_val_[key.get<std::string>()] = val;
            }
            return obj;
        } else if (*p == '[') {
            p++;
            json arr;
            arr.type_ = value_t::array;
            bool first = true;
            while (true) {
                while (p < end && std::isspace(*p)) p++;
                if (p >= end) throw std::runtime_error("unexpected end");
                if (*p == ']') {
                    p++;
                    break;
                }
                if (!first) {
                    if (*p != ',') throw std::runtime_error("expected comma");
                    p++;
                }
                first = false;
                json item = parse_impl(p, end);
                arr.array_val_.push_back(item);
            }
            return arr;
        } else if (*p == '"') {
            p++;
            std::string s;
            while (p < end && *p != '"') {
                if (*p == '\\') {
                    p++;
                    if (p >= end) throw std::runtime_error("unexpected end");
                    switch (*p) {
                        case 'n': s += '\n'; break;
                        case 'r': s += '\r'; break;
                        case 't': s += '\t'; break;
                        default: s += *p; break;
                    }
                } else {
                    s += *p;
                }
                p++;
            }
            if (p >= end) throw std::runtime_error("unexpected end");
            p++;
            return json(s);
        } else if (*p == 't' || *p == 'f') {
            if (p + 3 < end && *p == 't' && *(p+1) == 'r' && *(p+2) == 'u' && *(p+3) == 'e') {
                p += 4;
                return json(true);
            }
            if (p + 4 < end && *p == 'f' && *(p+1) == 'a' && *(p+2) == 'l' && *(p+3) == 's' && *(p+4) == 'e') {
                p += 5;
                return json(false);
            }
            throw std::runtime_error("invalid boolean");
        } else if (*p == 'n') {
            if (p + 3 < end && *p == 'n' && *(p+1) == 'u' && *(p+2) == 'l' && *(p+3) == 'l') {
                p += 4;
                return json(nullptr);
            }
            throw std::runtime_error("invalid null");
        } else if (*p == '-' || std::isdigit(*p)) {
            bool is_neg = (*p == '-');
            if (is_neg) p++;
            std::string num;
            bool is_float = false;
            while (p < end && (std::isdigit(*p) || *p == '.' || *p == 'e' || *p == 'E')) {
                if (*p == '.' || *p == 'e' || *p == 'E') is_float = true;
                num += *p;
                p++;
            }
            if (is_float) {
                double d = std::stod(num);
                return json(is_neg ? -d : d);
            } else {
                if (is_neg) {
                    return json(-std::stoll(num));
                } else {
                    return json(std::stoull(num));
                }
            }
        }
        throw std::runtime_error("invalid json");
    }

    static json parse(const char* start, const char* end) {
        const char* p = start;
        return parse_impl(p, end);
    }
};

inline std::istream& operator>>(std::istream& is, json& j) {
    std::string s((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    j = json::parse(s);
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const json& j) {
    os << j.dump();
    return os;
}

} // namespace nlohmann
