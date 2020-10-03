//
// Created by maxwe on 2020-10-02.
//

#ifndef CUDA_FUNCTIONS_CONTRACTS_H
#define CUDA_FUNCTIONS_CONTRACTS_H

#include <cassert>

#if defined(_MSC_VER) && !defined(__clang__)
#if defined(NDEBUG)
#define CU_assert_bool(Expr, Msg) true
#define CU_assert_result(Expr, Msg) Expr
#else
#define CU_assert_bool(Expr, Msg) (bool(Expr) || ((_wassert(L""Msg, L"" __FILE__, unsigned(__LINE__))), false))
#define CU_assert_result(Expr, Msg) CU_assert_bool(Expr, Msg)
#endif
#define CU_assert(Expr) (void)CU_assert_bool(Expr, #Expr)
#define CU_axiom(...) { bool __value_of_axiom(__VA_ARGS__); (void)CU_assert_bool(__value_of_axiom, #__VA_ARGS__); __assume(__value_of_axiom); }

#define CU_pre(...) CU_axiom(__VA_ARGS__)
#define CU_post(...) CU_axiom(__VA_ARGS__)
#elif defined(__clang__)
#if defined(NDEBUG)
#define CU_assert_bool(Expr, Msg) true
#define CU_assert_result(Expr, Msg) Expr
#else
#define CU_assert_bool(Expr, Msg) (bool(Expr) || ((_wassert(L""Msg, L"" __FILE__, unsigned(__LINE__))), false))
#define CU_assert_result(Expr, Msg) CU_assert_bool(Expr, Msg)
#endif
#define CU_assert(Expr) (void)CU_assert_bool(Expr, #Expr)
#define CU_axiom(...) { bool __value_of_axiom(__VA_ARGS__); (void)CU_assert_bool(__value_of_axiom, #__VA_ARGS__); __builtin_assume(__value_of_axiom); }

#elif defined(__GNUC__)

#endif

#endif//CUDA_FUNCTIONS_CONTRACTS_H
