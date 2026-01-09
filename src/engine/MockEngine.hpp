#pragma once

#include "../core/ISearchEngine.hpp"

class MockEngine : public ISearchEngine {
public:
    std::vector<SearchResult> search(
        const std::string& query,
        int limit = 50
    ) override;

    std::string name() const override;
};
