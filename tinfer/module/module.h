/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective
*contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <functional>
#include <string>
#include <string_view>

#include "boost/preprocessor/comparison/equal.hpp"
#include "boost/preprocessor/control/if.hpp"
#include "boost/preprocessor/library.hpp"
#include "boost/preprocessor/variadic/elem.hpp"
#include "boost/preprocessor/variadic/size.hpp"

#include "tinfer/module/context.h"
#include "tinfer/module/params.h"
#include "tinfer/tensor/tensor.h"

namespace tinfer {
class Module {
 public:
  using ParamSetFunc = std::function<void(const Params&)>;
  using ResetFunc = std::function<void(void)>;
  using FieldFunc = std::function<int(Context&)>;
  virtual std::string_view Name() const = 0;
  virtual std::string_view Desc() const = 0;
  virtual void Setup(Context& ctx) = 0;
  virtual void Forward(Context& ctx) = 0;

  int PreProcess(Context& ctx, Params& params);
  int PostProcess(Context& ctx);
  // Destructor.
  virtual ~Module() = default;

 protected:
  size_t RegisterParam(const std::string& name, const std::string& type, const std::string& deafult_value,
                       const std::string& desc, ParamSetFunc&& f);
  size_t RegisterInputTensor(const std::string& name, FieldFunc&& f);
  size_t RegisterOutputTensor(const std::string& name, FieldFunc&& f);

  size_t AddResetFunc(ResetFunc&& f);

  std::vector<ParamSetFunc> params_settings_;
  std::vector<ResetFunc> reset_funcs_;
  std::vector<FieldFunc> input_funcs_;
  std::vector<FieldFunc> output_funcs_;
};

typedef std::function<Module*(void)> ModuleCreator;

class ModuleFactory {
 public:
  static void Register(std::string_view name, std::string_view data_type, const ModuleCreator& creator);
  static Module* GetOperator(const std::string& name, const std::string& data_type);
};

struct ModuleRegister {
  ModuleRegister(std::string_view name, std::string_view data_type, const ModuleCreator& creator);
};

}  // namespace tinfer

#define MODULE_CLASS_NAME(...) BOOST_PP_CAT(Operator, BOOST_PP_CAT(BOOST_PP_VARIADIC_ELEM(0, __VA_ARGS__), Object))

#define MODULE_BEGIN(...)                                                                                           \
  namespace {                                                                                                       \
  namespace BOOST_PP_CAT(didagle_ops, __COUNTER__) {                                                                \
  struct MODULE_CLASS_NAME(__VA_ARGS__);                                                                            \
  using LocalProcessorClass = MODULE_CLASS_NAME(__VA_ARGS__);                                                       \
  using tinfer::Context;                                                                                            \
  using tinfer::TensorMap;                                                                                          \
  using tinfer::Tensor;                                                                                             \
  static const constexpr std::string_view k_local_processor_name =                                                  \
      BOOST_PP_STRINGIZE(BOOST_PP_VARIADIC_ELEM(0, __VA_ARGS__));                                                   \
  struct MODULE_CLASS_NAME(__VA_ARGS__) : public tinfer::Module {                                                   \
    static const constexpr std::string_view _local_processor_desc =                                                 \
        BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_VARIADIC_SIZE(__VA_ARGS__), 2), BOOST_PP_VARIADIC_ELEM(1, __VA_ARGS__), \
                    ("Empty Description"));                                                                         \
    std::string_view Name() const override { return k_local_processor_name; }                                       \
    std::string_view Desc() const override { return _local_processor_desc; }

#define MODULE_END                                                                              \
  }                                                                                             \
  ;                                                                                             \
  static tinfer::ModuleRegister BOOST_PP_CAT(instance_, __COUNTER__)(                           \
      k_local_processor_name, "", []() -> tinfer::Module* { return new LocalProcessorClass; }); \
  }  /* namespace BOOST_PP_CAT */                                                               \
  }  // namespace

#define GENERIC_MODULE_BEGIN(...)                                                                                   \
  namespace {                                                                                                       \
  namespace BOOST_PP_CAT(didagle_ops, __COUNTER__) {                                                                \
  template <typename T>                                                                                             \
  struct MODULE_CLASS_NAME(__VA_ARGS__);                                                                            \
  template <typename T>                                                                                             \
  using LocalProcessorClass = MODULE_CLASS_NAME(__VA_ARGS__)<T>;                                                    \
  using tinfer::Context;                                                                                            \
  using tinfer::TensorMap;                                                                                          \
  using tinfer::Tensor;                                                                                             \
  static const constexpr std::string_view k_local_processor_name =                                                  \
      BOOST_PP_STRINGIZE(BOOST_PP_VARIADIC_ELEM(0, __VA_ARGS__));                                                   \
  template <typename T>                                                                                             \
  struct MODULE_CLASS_NAME(__VA_ARGS__) : public tinfer::Module {                                                   \
    static const constexpr std::string_view _local_processor_desc =                                                 \
        BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_VARIADIC_SIZE(__VA_ARGS__), 2), BOOST_PP_VARIADIC_ELEM(1, __VA_ARGS__), \
                    ("Empty Description"));                                                                         \
    std::string_view Name() const override { return k_local_processor_name; }                                       \
    std::string_view Desc() const override { return _local_processor_desc; }

#define GENERIC_MODULE_END                                                                                 \
  }                                                                                                        \
  ;                                                                                                        \
  static tinfer::ModuleRegister BOOST_PP_CAT(instance_, __COUNTER__)(                                      \
      k_local_processor_name, "fp16", []() -> tinfer::Module* { return new LocalProcessorClass<half>; });  \
  static tinfer::ModuleRegister BOOST_PP_CAT(instance_, __COUNTER__)(                                      \
      k_local_processor_name, "fp32", []() -> tinfer::Module* { return new LocalProcessorClass<float>; }); \
  }  /* namespace BOOST_PP_CAT */                                                                          \
  }  // namespace

#define TENSOR_INPUT(NAME)                                                                 \
  ::tinfer::Tensor* NAME = nullptr;                                                        \
  size_t __input_##NAME##_code = RegisterInputTensor(#NAME, [this](tinfer::Context& ctx) { \
    this->NAME = ctx.GetTensor(#NAME);                                                     \
    return (!NAME) ? -1 : 0;                                                               \
  });

#define TENSOR_OUTPUT(NAME)                                                                 \
  ::tinfer::Tensor NAME = {};                                                               \
  size_t __input_##NAME##_code = RegisterOutputTensor(#NAME, [this](tinfer::Context& ctx) { \
    if (!NAME.Empty()) {                                                                    \
      ctx.InsertTenseor(#NAME, NAME);                                                       \
    }                                                                                       \
    return 0;                                                                               \
  });

#define PARAMS_string(name, val, txt)                                                                  \
  tinfer::ParamsString PARAMS_##name = val;                                                            \
  size_t __PARAMS_##name##_code =                                                                      \
      RegisterParam(BOOST_PP_STRINGIZE(name), "string", val, txt, [this](const tinfer::Params& args) { \
        if (args[BOOST_PP_STRINGIZE(name)].IsString()) {                                               \
          PARAMS_##name = args[BOOST_PP_STRINGIZE(name)].String();                                     \
        }                                                                                              \
      });                                                                                              \
  size_t __reset_PARAMS_##name##_code = AddResetFunc([this]() { PARAMS_##name = val; });

#define PARAMS_string_vector(name, val, txt)                                                                         \
  std::vector<tinfer::ParamsString> PARAMS_##name;                                                                   \
  size_t __PARAMS_##name##_code = RegisterParam(                                                                     \
      BOOST_PP_STRINGIZE(name), "vector<string>", BOOST_PP_STRINGIZE(val), txt, [this](const tinfer::Params& args) { \
        if (args.Contains(BOOST_PP_STRINGIZE(name))) {                                                               \
          PARAMS_##name.clear();                                                                                     \
          const auto& member = args[BOOST_PP_STRINGIZE(name)];                                                       \
          for (size_t i = 0; i < member.Size(); i++) {                                                               \
            PARAMS_##name.emplace_back(member[i].String());                                                          \
          }                                                                                                          \
        }                                                                                                            \
      });                                                                                                            \
  size_t __reset_PARAMS_##name##_code = AddResetFunc([this]() { PARAMS_##name = BOOST_PP_REMOVE_PARENS(val); });

#define PARAMS_int(name, val, txt)                                                                        \
  int64_t PARAMS_##name = val;                                                                            \
  size_t __PARAMS_##name##_code = RegisterParam(                                                          \
      BOOST_PP_STRINGIZE(name), "int", BOOST_PP_STRINGIZE(val), txt, [this](const tinfer::Params& args) { \
        if (args[BOOST_PP_STRINGIZE(name)].IsInt()) {                                                     \
          PARAMS_##name = args[BOOST_PP_STRINGIZE(name)].Int();                                           \
        }                                                                                                 \
      });                                                                                                 \
  size_t __reset_PARAMS_##name##_code = AddResetFunc([this]() { PARAMS_##name = val; });

#define PARAMS_int_vector(name, val, txt)                                                                         \
  std::vector<int64_t> PARAMS_##name;                                                                             \
  size_t __PARAMS_##name##_code = RegisterParam(                                                                  \
      BOOST_PP_STRINGIZE(name), "vector<int>", BOOST_PP_STRINGIZE(val), txt, [this](const tinfer::Params& args) { \
        if (args.Contains(BOOST_PP_STRINGIZE(name))) {                                                            \
          PARAMS_##name.clear();                                                                                  \
          const auto& member = args[BOOST_PP_STRINGIZE(name)];                                                    \
          for (size_t i = 0; i < member.Size(); i++) {                                                            \
            PARAMS_##name.emplace_back(member[i].Int());                                                          \
          }                                                                                                       \
        }                                                                                                         \
      });                                                                                                         \
  size_t __reset_PARAMS_##name##_code = AddResetFunc([this]() { PARAMS_##name = BOOST_PP_REMOVE_PARENS(val); });

#define PARAMS_bool(name, val, txt)                                                                        \
  bool PARAMS_##name = val;                                                                                \
  size_t __PARAMS_##name##_code = RegisterParam(                                                           \
      BOOST_PP_STRINGIZE(name), "bool", BOOST_PP_STRINGIZE(val), txt, [this](const tinfer::Params& args) { \
        if (args[BOOST_PP_STRINGIZE(name)].IsBool()) {                                                     \
          PARAMS_##name = args[BOOST_PP_STRINGIZE(name)].Bool();                                           \
        }                                                                                                  \
      });                                                                                                  \
  size_t __reset_PARAMS_##name##_code = AddResetFunc([this]() { PARAMS_##name = val; });

#define PARAMS_bool_vector(name, val, txt)                                                                        \
  std::vector<bool> PARAMS_##name;                                                                                \
  size_t __PARAMS_##name##_code = RegisterParam(                                                                  \
      BOOST_PP_STRINGIZE(name), "vector<int>", BOOST_PP_STRINGIZE(val), txt, [this](const tinfer::Params& args) { \
        if (args.Contains(BOOST_PP_STRINGIZE(name))) {                                                            \
          PARAMS_##name.clear();                                                                                  \
          const auto& member = args[BOOST_PP_STRINGIZE(name)];                                                    \
          for (size_t i = 0; i < member.Size(); i++) {                                                            \
            PARAMS_##name.emplace_back(member[i].Bool());                                                         \
          }                                                                                                       \
        }                                                                                                         \
      });                                                                                                         \
  size_t __reset_PARAMS_##name##_code = AddResetFunc([this]() { PARAMS_##name = BOOST_PP_REMOVE_PARENS(val); });

#define PARAMS_double(name, val, txt)                                                                        \
  double PARAMS_##name = val;                                                                                \
  size_t __PARAMS_##name##_code = RegisterParam(                                                             \
      BOOST_PP_STRINGIZE(name), "double", BOOST_PP_STRINGIZE(val), txt, [this](const tinfer::Params& args) { \
        if (args[BOOST_PP_STRINGIZE(name)].IsDouble()) {                                                     \
          PARAMS_##name = args[BOOST_PP_STRINGIZE(name)].Double();                                           \
        }                                                                                                    \
      });                                                                                                    \
  size_t __reset_PARAMS_##name##_code = AddResetFunc([this]() { PARAMS_##name = val; });

#define PARAMS_double_vector(name, val, txt)                                                                         \
  std::vector<double> PARAMS_##name;                                                                                 \
  size_t __PARAMS_##name##_code = RegisterParam(                                                                     \
      BOOST_PP_STRINGIZE(name), "vector<double>", BOOST_PP_STRINGIZE(val), txt, [this](const tinfer::Params& args) { \
        if (args.Contains(BOOST_PP_STRINGIZE(name))) {                                                               \
          PARAMS_##name.clear();                                                                                     \
          const auto& member = args[BOOST_PP_STRINGIZE(name)];                                                       \
          for (size_t i = 0; i < member.Size(); i++) {                                                               \
            PARAMS_##name.emplace_back(member[i].Double());                                                          \
          }                                                                                                          \
        }                                                                                                            \
      });                                                                                                            \
  size_t __reset_PARAMS_##name##_code = AddResetFunc([this]() { PARAMS_##name = BOOST_PP_REMOVE_PARENS(val); });
