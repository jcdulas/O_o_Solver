#include <oor/api/CardEdits.hpp>
#include <oor/api/ControlEdits.hpp>
#include <oor/base/text/Json.hpp>
#include <oos/api/ConstraintJacobian.hpp>
#include <oos/api/ModelFromDeck.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>

using Clock = std::chrono::steady_clock;
using Field = oor::model::FieldValue;

Field field(const oor::base::JsonValue& value)
{
    if (value.isNull())
        return {};
    if (value.isText())
        return {Field::Kind::Text, 0, 0., value.asText("field")};
    const auto n = value.asNumber("field");
    if (n == std::trunc(n))
        return {Field::Kind::Integer, static_cast<std::int64_t>(n), 0., {}};
    return {Field::Kind::Real, 0, n, {}};
}

struct Sink final : oos::design::ProductSink
{
    std::vector<double> values;
    void consume(const oos::design::ProductBlock& block) override
    {
        values.assign(block.values.begin(), block.values.end());
    }
};

int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;
    auto deck = oor::api::Deck::createModel();
    oor::api::setSolution(*deck, "LinearStatics");
    const oor::api::ControlRecord control[] = {{"SinglePointConstrSelection", {"1"}},
                                               {"DisplacementOutput", {"All"}},
                                               {"SubcaseStart", {"1"}},
                                               {"StaticLoadSelection", {"2"}},
                                               {"SubcaseDesignConstrSelection", {"7"}},
                                               {"DesignObjectiveSelection", {"11"}}};
    oor::api::setCaseControl(*deck, control);
    std::ifstream in(argv[1]);
    std::string text{std::istreambuf_iterator<char>(in), {}};
    const auto data = oor::base::JsonValue::parse(text);
    std::vector<oor::api::CardInput> cards;
    for (const auto& entry : data.asArray("cards"))
    {
        const auto& row = entry.asArray("card");
        oor::api::CardInput card{row.front().asText("type"), {}};
        std::transform(row.begin() + 1, row.end(), std::back_inserter(card.fields), field);
        cards.push_back(std::move(card));
    }
    oor::api::insertCards(*deck, cards);
    deck->applyDefaults();
    const auto start = Clock::now();
    const auto problem = oos::api::prepareOptimization(deck);
    const auto prepared = Clock::now();
    auto point = problem->evaluate(problem->problem().initialValues());
    const auto evaluated = Clock::now();
    const auto gradient = point->gradients();
    const auto differentiated = Clock::now();
    deck.reset();

    const std::vector<double> directions{1., 0., 0., 1.};
    Sink sink;
    auto product = point->jvp(directions, 2, 2, sink);
    const auto warm = Clock::now();
    for (int i = 0; i < 20; ++i)
        product = point->jvp(directions, 2, 2, sink);
    const auto end = Clock::now();
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::cout << "prepare_ms=" << ms(start, prepared) << " evaluate_ms=" << ms(prepared, evaluated)
              << " gradient_ms=" << ms(evaluated, differentiated) << " warm_jvp_ms=" << ms(warm, end) / 20
              << " gradient_factorisations=" << gradient.report.factorisations
              << " warm_jvp_factorisations=" << product.factorisations << " output_bytes=" << sink.values.size() * 8
              << '\n';

    oos::api::ConstraintJacobian csr(*point, std::vector<std::vector<std::size_t>>{{0, 1}, {0, 1}, {0}});
    std::vector<double> values(csr.columnIndices().size());
    auto second = problem->evaluate({3., 1000.});
    csr.values(*second, values);
    const auto reference = second->gradients();
    for (std::size_t r = 0; r < reference.constraints.size(); ++r)
        for (auto k = csr.rowOffsets()[r]; k < csr.rowOffsets()[r + 1]; ++k)
            if (std::abs(values[k] - reference.constraints[r][csr.columnIndices()[k]]) > 1e-9)
                return 1;
    const auto summary = point->optimizerValues();
    const std::vector<double> expected{-.0075, -.000015, 3., 0.};
    const auto close = [](double a, double b) { return std::abs(a - b) < 1e-8; };
    if (summary.partial || summary.uncertainMode || !summary.objective || !close(*summary.objective, 6.) ||
        !std::equal(sink.values.begin(), sink.values.end(), expected.begin(), expected.end(), close))
        return 1;
    return 0;
}
