#pragma once
#include "IFilter.hpp"

class ErrorDisclosureFilter : public IFilter {
public:
    std::string name() const override {
        return "error-disclosure";
    }

    bool match(const SearchResult& r) const override {
        return r.body.find("Fatal error") != std::string::npos ||
               r.body.find("Exception") != std::string::npos ||
               r.body.find("Stack trace") != std::string::npos;
    }
};
