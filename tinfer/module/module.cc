/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "tinfer/module/module.h"

#include "absl/hash/hash.h"

namespace tinfer {
size_t Module::RegisterParam(const std::string& name, const std::string& type, const std::string& deafult_value,
                             const std::string& desc, ParamSetFunc&& f) {
  params_settings_.emplace_back(f);
  return params_settings_.size();
}

size_t Module::AddResetFunc(ResetFunc&& f) {
  reset_funcs_.emplace_back(f);
  return reset_funcs_.size();
}

size_t Module::RegisterInputTensor(const std::string& name, FieldFunc&& f) {
  input_funcs_.emplace_back(std::move(f));
  return input_funcs_.size();
}
size_t Module::RegisterOutputTensor(const std::string& name, FieldFunc&& f) {
  output_funcs_.emplace_back(std::move(f));
  return output_funcs_.size();
}

int Module::PreProcess(Context& ctx, Params& params) {
  for (auto& f : params_settings_) {
    f(params);
  }
  for (auto& f : input_funcs_) {
    f(ctx);
  }
  return 0;
}
int Module::PostProcess(Context& ctx) {
  for (auto& f : output_funcs_) {
    f(ctx);
  }
  return 0;
}

struct ModuleRegistryKey {
  std::string name;
  std::string data_type;
  bool operator==(const ModuleRegistryKey& other) const { return name == other.name && data_type == other.data_type; }
};

struct ModuleRegistryKeyhash {
  size_t operator()(const ModuleRegistryKey& k) const { return absl::HashOf(k.name, k.data_type); }
};

using CreatorTable = absl::flat_hash_map<ModuleRegistryKey, ModuleCreator, ModuleRegistryKeyhash>;
static CreatorTable* g_creator_table = nullptr;

void delete_creator_table() { delete g_creator_table; }
CreatorTable& get_creator_table() {
  if (nullptr == g_creator_table) {
    g_creator_table = new CreatorTable;
    atexit(delete_creator_table);
  }
  return *g_creator_table;
}

void ModuleFactory::Register(std::string_view name, std::string_view data_type, const ModuleCreator& creator) {
  ModuleRegistryKey key;
  key.name.assign(name.data(), name.size());
  key.data_type.assign(data_type.data(), data_type.size());
  std::string name_str(name.data(), name.size());
  get_creator_table().emplace(std::make_pair(key, creator));
}
Module* ModuleFactory::GetOperator(const std::string& name, const std::string& data_type) {
  ModuleRegistryKey key;
  key.name = name;
  key.data_type = data_type;
  auto found = get_creator_table().find(key);
  if (found != get_creator_table().end()) {
    return found->second();
  }
  return nullptr;
}

ModuleRegister::ModuleRegister(std::string_view name, std::string_view data_type, const ModuleCreator& creator) {
  ModuleFactory::Register(name, data_type, creator);
}
}  // namespace tinfer