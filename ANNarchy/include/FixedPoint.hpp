/*
 *    FixedPoint.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2026  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    ANNarchy is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include <cstdint>
#include <type_traits>

/**
 * @brief   Template class for the usage of fixed-point types using power of 2 as scale factor.
 * @details Implements a format following the Q-format. Attention, we assume always the usage of a sign bit!
 */
template<int IntegerBits, int FractionBits>
class fixed_t{
    static_assert(IntegerBits > 0);
    static_assert(FractionBits >= 0);

    using storage_type =
        std::conditional_t<
            ((IntegerBits + FractionBits + 1) <= 8) ,  int8_t,
        std::conditional_t<
            ((IntegerBits + FractionBits + 1) <= 16), int16_t,
        std::conditional_t<
            ((IntegerBits + FractionBits + 1) <= 32), int32_t,
                               int64_t>>>;

    // scale-factor for conversion: 1 / 2**(FractionBits)
    static constexpr storage_type scale_ =
        storage_type(1) << FractionBits;

protected:
    storage_type value_;

public:

    /*
     *  constructors. ATTENTION: they scale the provided value automatically.
     */
    explicit constexpr fixed_t():
        fixed_t(static_cast<storage_type>(0)) {}

    explicit constexpr fixed_t(int value)
        : value_(value * scale_) {}

    explicit constexpr fixed_t(float value)
        : value_(value * scale_) {}

    explicit constexpr fixed_t(double value)
        : value_(value * scale_) {}

    explicit constexpr fixed_t(long int value)
        : value_(value * scale_) {}

    // Specialized constructor to initialize directly, e.g., used in the overloaded arithemtic operators.
    explicit constexpr fixed_t(storage_type value, bool)
        : value_(value) {}

    /*
     * arithmetic operators
     */
    fixed_t operator+(const fixed_t& other) const {
        return fixed_t(static_cast<storage_type>(value_ + other.value_), true);
    }

    fixed_t operator-(const fixed_t& other) const {
        return fixed_t(static_cast<storage_type>(value_ - other.value_), true);
    }

    fixed_t operator*(const fixed_t& other) const {
        int64_t result = (int64_t(value_) * int64_t(other.value_)) / (1LL << FractionBits);
        return fixed_t(static_cast<storage_type>(result), true);
    }

    fixed_t operator/(const fixed_t& other) const {
        int64_t numerator = int64_t(value_) * (1LL << FractionBits);
        return fixed_t(static_cast<storage_type>(numerator / int64_t(other.value_)), true);
    }

    /*
     * arithmetic operators with implicit type conversions
     */
    friend fixed_t operator+(double lhs, const fixed_t& rhs) {
        return fixed_t{lhs} + rhs;
    }

    friend fixed_t operator-(double lhs, const fixed_t& rhs) {
        return fixed_t{lhs} - rhs;
    }

    /*
     * in-place arithmetic operators
     */
    fixed_t& operator+=(const fixed_t& other) {
        value_ = static_cast<storage_type>(value_ + other.value_);
        return *this;
    }

    fixed_t& operator-=(const fixed_t& other) {
        value_ = static_cast<storage_type>(value_ - other.value_);
        return *this;
    }

    fixed_t& operator*=(const fixed_t& other) {
        int64_t result = (int64_t(value_) * int64_t(other.value_)) / (1LL << FractionBits);
        value_ = static_cast<storage_type>(result);
        return *this;
    }

    fixed_t& operator/=(const fixed_t& other) {
        int64_t result = (int64_t(value_) * (1LL << FractionBits)) / int64_t(other.value_);
        value_ = static_cast<storage_type>(result);
        return *this;
    }

    /*
     * comparison operators
     */
    bool operator>(const fixed_t& other) const {
        return value_ > other.value_;
    }

    bool operator<(const fixed_t& other) const {
        return value_ < other.value_;
    }

    bool operator==(const fixed_t& other) const {
        return value_ == other.value_;
    }

    bool operator!=(const fixed_t& other) const {
        return value_ != other.value_;
    }

    /*
     * comparison operators with implicit type conversions
     */
    bool operator>(const double& other) const {
        return value_ > fixed_t(other);
    }

    bool operator<(const double& other) const {
        return value_ < fixed_t(other);
    }

    /*
     * explicit type conversions (ordered by size in bits)
     */
    explicit operator int() const {
        return static_cast<int>(double(value_) / double(scale_));
    }

    explicit operator float() const {
        return static_cast<float>(double(value_) / double(scale_));
    }

    explicit operator double() const {
        return static_cast<double>(double(value_) / double(scale_));
    }

    explicit operator long int() const {
        return static_cast<long int>(double(value_) / double(scale_));
    }

    /*
     *  @brief      printout operator.
     *  @details    trigger conversion to double value and print out to output stream os.
     */
    friend std::ostream& operator<<(std::ostream& os, const fixed_t& obj) {
        return os << std::to_string(double{obj});
    }
};
