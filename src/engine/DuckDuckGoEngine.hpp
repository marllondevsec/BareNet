#pragma once

#include "../core/ISearchEngine.hpp"
#include <string>

class DuckDuckGoEngine : public ISearchEngine {
public:
    std::vector<SearchResult> search(
        const std::string& query,
        int limit = 20
    ) override;

    std::string name() const override;

private:
    std::string fetch(const std::string& url);
    std::vector<SearchResult> parseResults(const std::string& html, int limit);
};
