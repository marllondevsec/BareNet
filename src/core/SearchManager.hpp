#pragma once

#include <vector>
#include <memory>
#include <string>

#include "ISearchEngine.hpp"
#include "SearchResult.hpp"

class SearchManager {
public:
    // registra uma engine (Google, Bing, Mock, etc)
    void registerEngine(std::unique_ptr<ISearchEngine> engine);

    // executa busca em todas as engines
    std::vector<SearchResult> search(
        const std::string& query,
        int limit_per_engine = 50
    ) const;

private:
    std::vector<std::unique_ptr<ISearchEngine>> engines;

    // normalização de URL para deduplicação
    static std::string normalizeUrl(const std::string& url);
};
