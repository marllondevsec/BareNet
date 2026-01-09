#include "MockEngine.hpp"

std::vector<SearchResult> MockEngine::search(
    const std::string& query,
    int limit
) {
    std::vector<SearchResult> results;

    for (int i = 1; i <= limit; ++i) {
        SearchResult r;
        r.url = "http://example.com/result/" + std::to_string(i);
        r.title = "Mock result " + std::to_string(i);
        r.snippet = "Resultado simulado para query: " + query;
        r.engine = name();

        results.push_back(r);
    }

    return results;
}

std::string MockEngine::name() const {
    return "MockEngine";
}
