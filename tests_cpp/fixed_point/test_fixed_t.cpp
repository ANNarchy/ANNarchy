/*
 *    test_fixed_t.cpp
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
// Google's testbench (will be copied locally by CMake)
#include <gtest/gtest.h>

// ANNarchy's fixed-point arithmetic type
#include "FixedPoint.hpp"

template<int IntegerBits_, int FractionBits_>
struct BitWidthConfig
{
    static constexpr int IntegerBits = IntegerBits_;
    static constexpr int FractionBits = FractionBits_;
};

using TestConfigs = ::testing::Types<
    //BitWidthConfig<7,8>,
    BitWidthConfig<15,16>
>;

template<typename T>
class FixedPointTest: public ::testing::Test {};

TYPED_TEST_SUITE(FixedPointTest, TestConfigs);

/********************************************************************************************/
/*  Construction                                                                            */
/********************************************************************************************/
TYPED_TEST(FixedPointTest, ConstructFromInt) {
    auto value = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(4);

    EXPECT_EQ(int{value}, 4);
}

TYPED_TEST(FixedPointTest, ConstructFromFP32) {
    auto value = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(0.5f);

    EXPECT_EQ(float{value}, 0.5f);
}

TYPED_TEST(FixedPointTest, ConstructFromFP64) {
    auto value = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(0.5);

    EXPECT_EQ(double{value}, 0.5);
}

/********************************************************************************************/
/*  Addition of two values                                                                  */
/********************************************************************************************/
TYPED_TEST(FixedPointTest, AddOfInts) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);

    auto res = arg1 + arg2;

    EXPECT_EQ(int{res}, 10);
}

TYPED_TEST(FixedPointTest, AddOfFloats) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(0.5);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(3.5);

    auto res = arg1 + arg2;

    EXPECT_EQ(double{res}, 4.0);
}

TYPED_TEST(FixedPointTest, AddAndAssignOfInts) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);

    auto res = arg1;
    res += arg2;

    EXPECT_EQ(int{res}, 10);
}

TYPED_TEST(FixedPointTest, AddAndAssignOfFloats) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(0.5);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(3.5);

    auto res = arg1;
    res += arg2;

    EXPECT_EQ(double{res}, 4.0);
}

/********************************************************************************************/
/*  Substraction of two values                                                              */
/********************************************************************************************/
TYPED_TEST(FixedPointTest, SubOfInts) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);

    auto res = arg1 - arg2;

    EXPECT_EQ(int{res}, -6);
}

TYPED_TEST(FixedPointTest, SubAndAssignOfInts) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);

    auto res = arg1;
    res -= arg2;

    EXPECT_EQ(int{res}, -6);
}

/********************************************************************************************/
/*  Multiply of two values                                                                  */
/********************************************************************************************/
TYPED_TEST(FixedPointTest, MultOfInts) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);

    auto res = arg1 * arg2;

    EXPECT_EQ(int{res}, 16);
}

TYPED_TEST(FixedPointTest, MultOfFloats) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(0.5);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(4.0);

    auto res = arg1 * arg2;

    EXPECT_EQ(double{res}, 2.0);
}

TYPED_TEST(FixedPointTest, MultAndAssignOfInts) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);

    auto res = arg1;
    res *= arg2;

    EXPECT_EQ(int{res}, 16);
}

TYPED_TEST(FixedPointTest, MultAndAssignOfFloats) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(0.5);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(4.0);

    auto res = arg1;
    res *= arg2;

    EXPECT_EQ(double{res}, 2.0);
}

TYPED_TEST(FixedPointTest, MultOfIntsLNeg) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(-2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);

    auto res = arg1 * arg2;

    EXPECT_EQ(int{res}, -16);
}

TYPED_TEST(FixedPointTest, MultOfIntsRNeg) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(-8);

    auto res = arg1 * arg2;

    EXPECT_EQ(int{res}, -16);
}

TYPED_TEST(FixedPointTest, MultOfIntsBNeg) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(-2);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(-8);

    auto res = arg1 * arg2;

    EXPECT_EQ(int{res}, 16);
}

/********************************************************************************************/
/*  Dividing two values                                                                     */
/********************************************************************************************/
TYPED_TEST(FixedPointTest, DivOfInts) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);

    auto res = arg1 / arg2;

    EXPECT_EQ(int{res}, 4);
}

TYPED_TEST(FixedPointTest, DivOfFloats) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(0.5);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(4.0);

    auto res = arg1 / arg2;

    EXPECT_EQ(double{res}, 0.125);
}

TYPED_TEST(FixedPointTest, DivAndAssignOfInts) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);

    auto res = arg1;
    res /= arg2;

    EXPECT_EQ(int{res}, 4.0);
}

TYPED_TEST(FixedPointTest, DivAndAssignOfFloats) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(0.5);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(4.0);

    auto res = arg1;
    res /= arg2;

    EXPECT_EQ(double{res}, 0.125);
}

TYPED_TEST(FixedPointTest, DivOfIntsLNeg) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(-8);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(2);

    auto res = arg1 / arg2;

    EXPECT_EQ(int{res}, -4);
}

TYPED_TEST(FixedPointTest, DivOfIntsRNeg) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(8);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(-2);

    auto res = arg1 / arg2;

    EXPECT_EQ(int{res}, -4);
}

TYPED_TEST(FixedPointTest, DivOfIntsBNeg) {
    auto arg1 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(-8);
    auto arg2 = fixed_t<TypeParam::IntegerBits, TypeParam::FractionBits>(-2);

    auto res = arg1 / arg2;

    EXPECT_EQ(int{res}, 4);
}
