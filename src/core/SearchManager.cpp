#include "SearchManager.hpp"

#include <unordered_set>
#include <algorithm>

void SearchManager::registerEngine(std::unique_ptr<ISearchEngine> engine)
{
    engines.push_back(std::move(engine));
}

std::vector<SearchResult> SearchManager::search(
    const std::string& query,
    int limit_per_engine
) const
{
    std::vector<SearchResult> all_results;
    std::unordered_set<std::string> seen_urls;

    for (const auto& engine : engines) {
        auto results = engine->search(query, limit_per_engine);

        for (auto& r : results) {
            std::string normalized = normalizeUrl(r.url);

            if (seen_urls.insert(normalized).second) {
                all_results.push_back(std::move(r));
            }
        }
    }

    return all_results;
}

std::string SearchManager::normalizeUrl(const std::string& url)
{
    std::string out = url;

    // lowercase
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);

    // remove protocolo
    if (out.rfind("http://", 0) == 0)
        out.erase(0, 7);
    else if (out.rfind("https://", 0) == 0)
        out.erase(0, 8);

    // remove www.
    if (out.rfind("www.", 0) == 0)
        out.erase(0, 4);

    // remove barra final
    if (!out.empty() && out.back() == '/')
        out.pop_back();

    return out;
}
