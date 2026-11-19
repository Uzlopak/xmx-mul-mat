// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xmx {
enum class Type { q4_k, q6_k, q8_0, q2_k, q5_k, q3_k };
enum class Glu { swiglu, geglu };
Type parse_type(const std::string & name);
const char * type_name(Type type);
float half_to_float(uint16_t bits);
uint16_t float_to_half(float value);

struct Fixture {
    Type type;
    int rows;
    int columns;
    int k;
    int activation_stride;
    std::vector<uint8_t> weights;
    std::vector<uint8_t> activations;
    std::vector<float> decoded_weights;
    std::vector<float> decoded_activations;
    std::vector<float> expected;
};

Fixture make_fixture(Type type, int rows, int columns, int k, unsigned seed = 42);
Fixture make_fixture_with_varying_scales(Type type, int rows, int columns, int k, unsigned seed = 42);
std::vector<float> glu_reference(const Fixture & up, const Fixture & gate, Glu operation);
struct Comparison {
    bool passed;
    double relative_l2;
    double max_absolute;
};
Comparison compare(const std::vector<float> & actual, const std::vector<float> & expected);
}
