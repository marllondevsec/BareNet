#pragma once

#include <string>
#include <vector>
#include "SearchResult.hpp"

class ISearchEngine {
public:
    virtual ~ISearchEngine() = default;

    // Executa uma busca passiva
    virtual std::vector<SearchResult> search(
        const std::string& query,
        int limit = 50
    ) = 0;

    // Nome da engine (exibição no CLI)
    virtual std::string name() const = 0;
};
