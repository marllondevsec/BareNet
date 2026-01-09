#pragma once

#include <string>
#include <vector>

class Screen {
public:
    static void clear();

    static void render(
        const std::string& title,
        const std::vector<std::string>& content,
        const std::vector<std::string>& menu
    );
};
