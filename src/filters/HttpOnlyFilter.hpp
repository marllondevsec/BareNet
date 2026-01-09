#pragma once
#include "IFilter.hpp"

class HttpOnlyFilter : public IFilter {
public:
    std::string name() const override {
        return "http-only";
    }

    bool match(const SearchResult& r) const override {
        return r.url.rfind("http://", 0) == 0;
    }
};
