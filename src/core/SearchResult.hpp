#pragma once
#include <string>

struct SearchResult {
    std::string url;
    std::string title;
    std::string body;
    int status_code;
};
