#pragma once
#include <string>
#include "../core/SearchResult.hpp"

class IFilter {
public:
    virtual ~IFilter() = default;
    virtual std::string name() const = 0;
    virtual bool match(const SearchResult& result) const = 0;
};
