#pragma once

#include <string>
#include <stdexcept>

namespace aurelis {

enum class ErrorCode {
    Ok = 0,
    FileNotFound,
    InvalidJson,
    InvalidConfig,
    TensorInvalidShape,
    TensorNotFinite,
    CheckpointMagicMismatch,
    CheckpointVersionMismatch
};

class AurelisException : public std::runtime_error {
public:
    ErrorCode code;

    AurelisException(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code(code) {}

    static const char* codeToString(ErrorCode code) {
        switch (code) {
            case ErrorCode::Ok: return "Ok";
            case ErrorCode::FileNotFound: return "FileNotFound";
            case ErrorCode::InvalidJson: return "InvalidJson";
            case ErrorCode::InvalidConfig: return "InvalidConfig";
            case ErrorCode::TensorInvalidShape: return "TensorInvalidShape";
            case ErrorCode::TensorNotFinite: return "TensorNotFinite";
            case ErrorCode::CheckpointMagicMismatch: return "CheckpointMagicMismatch";
            case ErrorCode::CheckpointVersionMismatch: return "CheckpointVersionMismatch";
            default: return "UnknownError";
        }
    }
};

} // namespace aurelis
