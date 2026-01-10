#include "FilterManager.hpp"

void FilterManager::registerFilter(std::unique_ptr<IFilter> filter)
{
    filters[filter->name()] = std::move(filter);
}

void FilterManager::enable(const std::string& name)
{
    if (filters.count(name))
        active.insert(name);
}

void FilterManager::disable(const std::string& name)
{
    active.erase(name);
}

bool FilterManager::isActive(const std::string& name) const
{
    return active.count(name);
}

std::vector<std::string> FilterManager::list() const
{
    std::vector<std::string> names;
    for (const auto& [name, _] : filters)
        names.push_back(name);
    return names;
}

std::vector<SearchResult> FilterManager::apply(
    const std::vector<SearchResult>& results
) const
{
    std::vector<SearchResult> out;

    for (const auto& r : results) {
        bool ok = true;
        for (const auto& name : active) {
            if (!filters.at(name)->match(r)) {
                ok = false;
                break;
            }
        }
        if (ok)
            out.push_back(r);
    }
    return out;
}
