#include "Cli.hpp"

#include <iostream>
#include <memory>

#include "../ui/Screen.hpp"

#include "../filters/ErrorDisclosureFilter.hpp"
#include "../filters/HttpOnlyFilter.hpp"
#include "../filters/ExposedApiFilter.hpp"

Cli::Cli()
{
    filters.registerFilter(std::make_unique<ErrorDisclosureFilter>());
    filters.registerFilter(std::make_unique<HttpOnlyFilter>());
    filters.registerFilter(std::make_unique<ExposedApiFilter>());
}

void Cli::run()
{
    std::string input;

    while (running) {
        showMainMenu();
        std::getline(std::cin, input);
        handleMainInput(input);
    }
}

void Cli::showMainMenu()
{
    Screen::render(
        "Menu Principal",
        {},
        {
            "[1] Filtros",
            "[2] Pesquisar",
            "[0] Sair"
        }
    );

    std::cout << "\nbarenet > ";
}

void Cli::handleMainInput(const std::string& input)
{
    if (input == "1") {
        filtersMenu();
    }
    else if (input == "2") {
        searchScreen();
    }
    else if (input == "0") {
        running = false;
    }
}

void Cli::filtersMenu()
{
    std::string input;

    while (true) {
        std::vector<std::string> body;
        auto list = filters.list();

        int index = 1;
        for (const auto& f : list) {
            body.push_back(
                "[" + std::to_string(index++) + "] " +
                f + " " +
                (filters.isActive(f) ? "[x]" : "[ ]")
            );
        }

        Screen::render(
            "Filtros",
            body,
            {
                "[0] Voltar",
                "Selecione um número para alternar"
            }
        );

        std::cout << "\n> ";
        std::getline(std::cin, input);

        if (input == "0")
            break;

        try {
            int idx = std::stoi(input) - 1;
            if (idx >= 0 && idx < (int)list.size()) {
                if (filters.isActive(list[idx]))
                    filters.disable(list[idx]);
                else
                    filters.enable(list[idx]);
            }
        }
        catch (...) {
            // entrada inválida → ignora silenciosamente
        }
    }
}

void Cli::searchScreen()
{
    Screen::render(
        "Pesquisar",
        {
            "[!] Engine ainda não implementada"
        },
        {
            "[0] Voltar"
        }
    );

    std::string _;
    std::getline(std::cin, _);
}
