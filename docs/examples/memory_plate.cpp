#include <oor/api/CardEdits.hpp>
#include <oor/api/ControlEdits.hpp>
#include <oos/api/Analyses.hpp>
#include <oos/api/ModelFromDeck.hpp>

#include <cmath>
#include <iostream>
#include <map>

using Field = oor::model::FieldValue;
Field integer(int value) { return {Field::Kind::Integer, value, 0., {}}; }
Field real(double value) { return {Field::Kind::Real, 0, value, {}}; }

int main()
{
    auto deck = oor::api::Deck::createModel();
    oor::api::setSolution(*deck, "LinearStatics");
    const oor::api::ControlRecord controls[] = {
        {"SinglePointConstrSelection", {"7"}}, {"DisplacementOutput", {"All"}},
        {"SubcaseStart", {"1"}}, {"StaticLoadSelection", {"10"}}
    };
    oor::api::setCaseControl(*deck, controls);
    const oor::api::CardInput cards[] = {
        {"Node", {integer(1), {}, real(0), real(0), real(0)}},
        {"Node", {integer(2), {}, real(1), real(0), real(0)}},
        {"Node", {integer(3), {}, real(1), real(1), real(0)}},
        {"Node", {integer(4), {}, real(0), real(1), real(0)}},
        {"ShellQuad4", {integer(1), integer(1), integer(1), integer(2), integer(3), integer(4)}},
        {"ShellProp", {integer(1), integer(1), real(0.01)}},
        {"IsoMat", {integer(1), real(100000), {}, real(0)}},
        {"SinglePointConstrList", {integer(7), integer(123456), integer(1), integer(4)}},
        {"PointForceLoad", {integer(10), integer(2), integer(0), real(.5), real(1), real(0), real(0)}},
        {"PointForceLoad", {integer(10), integer(3), integer(0), real(.5), real(1), real(0), real(0)}}
    };
    oor::api::insertCards(*deck, cards);
    deck->applyDefaults();
    const auto problems = deck->checkReferences();
    if (!problems.empty()) {
        std::cerr << problems.front().message << '\n';
        return 1;
    }
    const auto model = oos::api::modelFromDeck(deck);
    const auto run = oos::api::runAnalysis(*model, oos::api::Analysis::Statics, {});
    std::map<int, double> axial;
    for (const auto& table : run.tables.nodalTables)
        if (table.kind == "displacements")
            for (std::size_t row = 0; row < table.rowCount(); ++row)
                axial.emplace(table.ids[row], table.values[6 * row]);

    for (const auto& [node, displacement] : axial)
        std::cout << node << " " << displacement << '\n';
    return axial.size() == 4 && std::abs(axial.at(1)) < 1e-14 &&
           std::abs(axial.at(2) - .001) < 1e-13 && std::abs(axial.at(3) - .001) < 1e-13 &&
           std::abs(axial.at(4)) < 1e-14 ? 0 : 1;
}
