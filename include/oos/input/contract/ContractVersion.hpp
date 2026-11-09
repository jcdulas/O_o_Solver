#pragma once

#include "oos/base/arrow/ImportedTable.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace oos::input
{

// The reader contracts this solver has been built and tested against. Contract 75 uses descriptive element types,
// identical in model records and result containers.
inline constexpr std::array<std::int64_t, 1> ACCEPTED_DATA_CONTRACT_VERSIONS{75};

// The first contract whose grid point force tables may carry a multipoint constraint force row per relation and grid
// point: a reader of an earlier contract would take a split row for an element's row and drop its force from a cut, so
// below it the rows stay one per grid point. Every accepted contract is later; a table built by hand (version 0) is
// not.
inline constexpr std::int64_t FIRST_CONTRACT_WITH_SPLIT_MPC_ROWS = 56;

// The first contract whose tables keep a blank field blank (NaN) where no number means what the blank means; before it
// a blank transverse shear modulus of an orthotropic material was 1.0E8 (see ShellMaterial.cpp).
inline constexpr std::int64_t FIRST_CONTRACT_WITH_BLANK_FIELDS = 45;

// The key under which O_o_Reader writes its contract version in the Arrow schema metadata of every table it exports.
inline constexpr std::string_view DATA_CONTRACT_VERSION_KEY = "oor_reader.data_contract_version";

[[nodiscard]] bool isAcceptedContractVersion(std::int64_t version) noexcept;

// The accepted versions as a message prints them: "{21}" or "{20, 21}".
[[nodiscard]] std::string acceptedContractVersionsText();

// Throws ContractError when `version` is not accepted and not newer than the last accepted one (a newer contract is
// read). `where` names what carries the version ("the installed O_o_Reader", "table 'singlePointConstrTable'"); the
// message gives both versions and what to install.
void requireAcceptedContractVersion(std::int64_t version, std::string_view where);

// Checks the version written in the metadata of a table, when there is one, and returns it (0 when the table carries
// none: a table built by the user or by a test, which only its column contract vouches for). Throws ContractError for
// a version that is not accepted or not a number.
std::int64_t requireAcceptedTableVersion(const base::ImportedTable& table, std::string_view tableName);

} // namespace oos::input
