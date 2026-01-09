#pragma once
#include "IFilter.hpp"

class ExposedApiFilter : public IFilter {
public:
    std::string name() const override {
        return "exposed-api";
    }

    bool match(const SearchResult& r) const override {
        return r.url.find("/api") != std::string::npos ||
               r.body.find("swagger") != std::string::npos ||
               r.body.find("openapi") != std::string::npos;
    }
};
