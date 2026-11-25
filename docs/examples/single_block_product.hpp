#pragma once

#include <oos/api/Optimization.hpp>

#include <stdexcept>
#include <vector>

// For examples requesting the entire product in one block (block size >= count).
struct SingleBlockProduct final : oos::design::ProductSink
{
    std::vector<double> values;
    void consume(const oos::design::ProductBlock& block) override
    {
        if (block.first != 0)
            throw std::logic_error("expected a single product block");
        values.assign(block.values.begin(), block.values.end());
    }
};
