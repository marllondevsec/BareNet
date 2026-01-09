#pragma once
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "IFilter.hpp"

class FilterManager {
public:
    void registerFilter(std::unique_ptr<IFilter> filter);

    void enable(const std::string& name);
    void disable(const std::string& name);

    bool isActive(const std::string& name) const;
    std::vector<std::string> list() const;

    std::vector<SearchResult> apply(
        const std::vector<SearchResult>& results
    ) const;

private:
    std::unordered_map<std::string, std::unique_ptr<IFilter>> filters;
    std::unordered_set<std::string> active;
};
