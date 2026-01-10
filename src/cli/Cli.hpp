#pragma once

#include "../core/SearchManager.hpp"
#include "../filters/FilterManager.hpp" // <--- ESSENCIAL

#include <string>

class Cli {
public:
    Cli();
    void run();

private:
    void showMainMenu();
    void handleMainInput(const std::string& input);
    void filtersMenu();
    void searchScreen();

    bool running = true;

    SearchManager searchManager;
    FilterManager filters; // <--- agora ele conhece a classe
};
