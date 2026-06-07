#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "aurelis/tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Aurelis core bindings";
    py::class_<Tensor>(m, "Tensor")
        .def(py::init([](const std::vector<int64_t>& shape) {
            return Tensor::zeros(shape);
        }))
        .def("numel", &Tensor::numel)
        .def("shape", [](const Tensor& self) {
            return self.shape();
        })
        .def("__getitem__", [](const Tensor& self, int idx) {
            return self.data()[idx];
        })
        .def("__setitem__", [](Tensor& self, int idx, float val) {
            self.data()[idx] = val;
        })
        .def("to_numpy", [](const Tensor& self) {
            return py::array_t<float>(
                self.shape(),
                self.shape(),  // strides
                self.data()
            );
        });
}
