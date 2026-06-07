#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "aurelis/tensor.hpp"
#include "aurelis/lens/config.hpp"
#include "aurelis/config.hpp"

namespace py = pybind11;
namespace al = aurelis::lens;
using namespace aurelis;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Aurelis core bindings";

    // Tensor
    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](const std::vector<int64_t>& shape) {
            return Tensor::zeros(shape);
        }))
        .def_static("zeros", [](const std::vector<int64_t>& shape) {
            return Tensor::zeros(shape);
        })
        .def_static("from_data", [](const std::vector<int64_t>& shape, py::array_t<float> data) {
            return Tensor::from_data(shape, static_cast<const float*>(data.data()));
        })
        .def("numel", &Tensor::numel)
        .def("ndim", &Tensor::ndim)
        .def("shape", [](const Tensor& self) {
            return self.shape();
        })
        .def("__getitem__", [](const Tensor& self, int idx) {
            return self.at(idx);
        })
        .def("__setitem__", [](Tensor& self, int idx, float val) {
            self.at(idx) = val;
        })
        .def("to_numpy", [](const Tensor& self) {
            return py::array_t<float>(
                self.shape(),
                self.data()
            );
        })
        .def("save", &Tensor::save)
        .def_static("load", &Tensor::load);

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

    // Error handling
    py::register_exception<AurelisException>(m, "AurelisException");
}
