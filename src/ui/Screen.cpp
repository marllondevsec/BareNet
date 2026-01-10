#include "Screen.hpp"
#include "Banner.hpp"

#include <iostream>

void Screen::clear()
{
    // Limpa tela e move cursor para o topo (ANSI)
    std::cout << "\033[2J\033[H";
}

void Screen::render(
    const std::string& title,
    const std::vector<std::string>& content,
    const std::vector<std::string>& menu
)
{
    clear();
    Banner::render();

    if (!title.empty()) {
        std::cout << "\n--- " << title << " ---\n\n";
    }

    for (const auto& line : content) {
        std::cout << " " << line << "\n";
    }

    if (!content.empty())
        std::cout << "\n";

    for (const auto& item : menu) {
        std::cout << " " << item << "\n";
    }
}
