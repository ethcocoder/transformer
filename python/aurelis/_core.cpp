#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "aurelis/tensor.hpp"
#include "aurelis/lens/linear.hpp"
#include "aurelis/lens/config.hpp"
#include "aurelis/config.hpp"

namespace py = pybind11;
using namespace aurelis;
using namespace aurelis::lens;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Project Aurelis - O(n) Generative Substrate Core Bindings";

    // Error Code Enum
    py::enum_<ErrorCode>(m, "ErrorCode")
        .value("OK", ErrorCode::Ok)
        .value("FILE_NOT_FOUND", ErrorCode::FileNotFound)
        .value("INVALID_JSON", ErrorCode::InvalidJson)
        .value("INVALID_CONFIG", ErrorCode::InvalidConfig)
        .value("TENSOR_INVALID_SHAPE", ErrorCode::TensorInvalidShape)
        .value("TENSOR_NOT_FINITE", ErrorCode::TensorNotFinite)
        .value("CHECKPOINT_MAGIC_MISMATCH", ErrorCode::CheckpointMagicMismatch)
        .value("CHECKPOINT_VERSION_MISMATCH", ErrorCode::CheckpointVersionMismatch)
        .export_values();

    // Aurelis Exception
    py::register_exception<AurelisException>(m, "AurelisException");

    // Tensor Class
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int64_t>, bool>(), 
             py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("zeros", &Tensor::zeros, 
                    py::arg("shape"), py::arg("requires_grad") = false)
        .def_static("from_data", [](const std::vector<int64_t>& shape, 
                                    py::array_t<float> data, 
                                    bool requires_grad) {
            return Tensor::from_data(shape, static_cast<const float*>(data.data()), requires_grad);
        }, py::arg("shape"), py::arg("data"), py::arg("requires_grad") = false)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def("__getitem__", [](const Tensor& self, int64_t i) {
            return self.at(i);
        })
        .def("__setitem__", [](Tensor& self, int64_t i, float v) {
            self.at(i) = v;
        })
        .def("view", &Tensor::view)
        .def("clone", &Tensor::clone)
        .def("fill", &Tensor::fill)
        .def("save", &Tensor::save)
        .def_static("load", &Tensor::load)
        .def("to_numpy", [](const Tensor& self) {
            return py::array_t<float>(self.shape(), self.data());
        });

    // Linear Layer
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int>(), py::arg("in_features"), py::arg("out_features"))
        .def("forward", &Linear::forward)
        .def("backward", [](const Linear& self, const Tensor& grad_out, const Tensor& x) {
            Tensor grad_W(self.weight().shape());
            Tensor grad_b(self.bias().shape());
            Tensor grad_x({x.shape()[0], self.in_features()});
            self.backward(grad_out, x, grad_W, grad_b, grad_x);
            return std::make_tuple(grad_W, grad_b, grad_x);
        })
        .def("init_xavier", &Linear::init_xavier)
        .def_property_readonly("weight", py::overload_cast<>(&Linear::weight))
        .def_property_readonly("bias", py::overload_cast<>(&Linear::bias))
        .def_property_readonly("in_features", &Linear::in_features)
        .def_property_readonly("out_features", &Linear::out_features)
        .def("save", &Linear::save)
        .def_static("load", &Linear::load);

    // Tensor Operations
    m.def("add", [](const Tensor& a, const Tensor& b) {
        Tensor out(a.shape());
        add(a, b, out);
        return out;
    }, "Add two tensors element-wise");
    m.def("mul", [](const Tensor& a, const Tensor& b) {
        Tensor out(a.shape());
        mul(a, b, out);
        return out;
    }, "Multiply two tensors element-wise");
    m.def("matmul", [](const Tensor& A, const Tensor& B) {
        int64_t m = A.shape()[0];
        int64_t k = A.shape()[1];
        int64_t n = B.shape()[1];
        Tensor out({m, n});
        matmul(A, B, out);
        return out;
    }, "Matrix multiplication of two tensors");

    // Configs
    py::class_<LensConfig>(m, "LensConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &LensConfig::vocab_size)
        .def_readwrite("D", &LensConfig::D)
        .def_readwrite("d_model", &LensConfig::d_model)
        .def_readwrite("d_tau", &LensConfig::d_tau)
        .def_readwrite("d_ff", &LensConfig::d_ff)
        .def_readwrite("num_layers", &LensConfig::num_layers)
        .def_readwrite("num_scales", &LensConfig::num_scales)
        .def_readwrite("lambda_stab", &LensConfig::lambda_stab)
        .def_readwrite("lambda_aux", &LensConfig::lambda_aux)
        .def_readwrite("lr", &LensConfig::lr);

    py::class_<AurelisConfig>(m, "AurelisConfig")
        .def(py::init<>())
        .def_readwrite("lens", &AurelisConfig::lens)
        .def("save", &AurelisConfig::save)
        .def_static("load", &AurelisConfig::load);
}
